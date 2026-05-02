"""
Training Pipeline for the Transformer v2 MCM Split Network.
=============================================================
4-stage curriculum learning: tiny → short → medium → full chains.
Identical training strategy to the Pointer Network for fair comparison.

Architecture difference: Cross-attention decoder (multi-head dot-product)
instead of Pointer Network's Bahdanau (additive) attention.

Usage:
    python -m src.train_transformer_v2
    python -m src.train_transformer_v2 --stage 1      # run only stage 1
    python -m src.train_transformer_v2 --resume models/transformer_v2_stage2.pth --stage 3
"""

import torch
import torch.optim as optim
import os
import time
import argparse
import numpy as np
from tqdm import tqdm

from src.models.transformer_split import TransformerMCMSplitNet
from src.models.pointer_mcm import compute_cost_from_splits
from src.data.pointer_loader import create_pointer_dataloaders
from src.utils.pointer_losses import PointerNetLoss


# ─── Configuration ──────────────────────────────────────────────────────

CURRICULUM = {
    1: {'max_n': 5,  'epochs': 30,  'lr': 3e-4, 'label': 'Tiny (n<=5)'},
    2: {'max_n': 15, 'epochs': 50,  'lr': 2e-4, 'label': 'Short (n<=15)'},
    3: {'max_n': 30, 'epochs': 80,  'lr': 1e-4, 'label': 'Medium (n<=30)'},
    4: {'max_n': 50, 'epochs': 140, 'lr': 5e-5, 'label': 'Full (n<=50)'},
}

DATA_PATH = 'data/mcm_120000.json'
MODEL_DIR = 'models'
BATCH_SIZE = 128
PATIENCE = 20
GRAD_CLIP = 1.0

MODEL_PREFIX = 'transformer_v2'


def setup_device():
    """Detect GPU/CPU and print info."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  CUDA: {torch.version.cuda} | "
              f"Memory: {props.total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("WARNING: No GPU detected. Training on CPU (will be slow).")
    return device


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    """Train for one epoch. Returns avg loss, split accuracy."""
    model.train()
    total_loss = 0
    total_s_loss = 0
    total_c_loss = 0
    total_acc = 0
    n_batches = 0

    for batch in tqdm(loader, leave=False, desc="Train Batch"):
        seq_feats, pad_mask, split_tgt, split_mask, cost_tgt, actual_n, raw_dims = batch
        seq_feats = seq_feats.to(device)
        pad_mask = pad_mask.to(device)
        split_tgt = split_tgt.to(device)
        cost_tgt = cost_tgt.to(device)
        actual_n = actual_n.to(device)

        optimizer.zero_grad()

        # Forward
        split_logits, split_valid, aux_cost = model(seq_feats, pad_mask, actual_n)

        # Loss (same PointerNetLoss — architecture-agnostic)
        loss, s_loss, c_loss, s_acc = criterion(
            split_logits, split_valid, split_tgt, actual_n, aux_cost, cost_tgt
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_s_loss += s_loss
        total_c_loss += c_loss
        total_acc += s_acc
        n_batches += 1

    return (total_loss / n_batches, total_s_loss / n_batches,
            total_c_loss / n_batches, total_acc / n_batches)


def validate(model, loader, criterion, device):
    """Validate. Returns avg loss, split accuracy."""
    model.eval()
    total_loss = 0
    total_s_loss = 0
    total_c_loss = 0
    total_acc = 0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="Val Batch"):
            seq_feats, pad_mask, split_tgt, split_mask, cost_tgt, actual_n, raw_dims = batch
            seq_feats = seq_feats.to(device)
            pad_mask = pad_mask.to(device)
            split_tgt = split_tgt.to(device)
            cost_tgt = cost_tgt.to(device)
            actual_n = actual_n.to(device)

            split_logits, split_valid, aux_cost = model(seq_feats, pad_mask, actual_n)
            loss, s_loss, c_loss, s_acc = criterion(
                split_logits, split_valid, split_tgt, actual_n, aux_cost, cost_tgt
            )

            total_loss += loss.item()
            total_s_loss += s_loss
            total_c_loss += c_loss
            total_acc += s_acc
            n_batches += 1

    return (total_loss / max(n_batches, 1), total_s_loss / max(n_batches, 1),
            total_c_loss / max(n_batches, 1), total_acc / max(n_batches, 1))


def evaluate_cost_accuracy(model, loader, device, max_samples=2000):
    """
    Evaluate the actual cost accuracy by computing costs from predicted splits.
    This is the TRUE metric — MAPE between predicted cost and DP cost.
    """
    model.eval()
    all_pred_costs = []
    all_true_costs = []
    all_split_accs = []
    count = 0

    with torch.no_grad():
        for batch in loader:
            seq_feats, pad_mask, split_tgt, split_mask, cost_tgt, actual_n, raw_dims = batch
            seq_feats = seq_feats.to(device)
            pad_mask = pad_mask.to(device)
            actual_n = actual_n.to(device)

            predicted_splits, aux_cost = model.predict(seq_feats, pad_mask, actual_n)

            for b in range(len(raw_dims)):
                dims = raw_dims[b]
                n = actual_n[b].item()
                true_cost = np.expm1(cost_tgt[b].item())

                # Compute cost from predicted splits
                pred_cost = compute_cost_from_splits(dims, predicted_splits[b])
                all_pred_costs.append(pred_cost)
                all_true_costs.append(true_cost)

                # Check split accuracy (skip L=2 trivial splits)
                correct = 0
                total = 0
                for i in range(1, n + 1):
                    for j in range(i + 2, n + 1):  # j >= i+2 means L >= 3
                        true_k = split_tgt[b, i - 1, j - 1].item() + i
                        pred_k = predicted_splits[b].get((i, j), -1)
                        if pred_k == true_k:
                            correct += 1
                        total += 1
                if total > 0:
                    all_split_accs.append(correct / total)

                count += 1
                if count >= max_samples:
                    break
            if count >= max_samples:
                break

    pred_arr = np.array(all_pred_costs, dtype=np.float64)
    true_arr = np.array(all_true_costs, dtype=np.float64)

    # MAPE
    mape = np.mean(np.abs(true_arr - pred_arr) / (true_arr + 1e-8)) * 100

    # Exact match rate (cost within 0.1%)
    exact_rate = np.mean(np.abs(true_arr - pred_arr) / (true_arr + 1e-8) < 0.001) * 100

    # Failure rate (cost error > 1%)
    failure_rate = np.mean(np.abs(true_arr - pred_arr) / (true_arr + 1e-8) > 0.01) * 100

    # Average split accuracy
    avg_split_acc = np.mean(all_split_accs) * 100 if all_split_accs else 0

    return mape, exact_rate, avg_split_acc, failure_rate


def train_stage(model, stage, device, resume_path=None):
    """Train one curriculum stage."""
    cfg = CURRICULUM[stage]
    print(f"\n{'=' * 70}")
    print(f"  TRANSFORMER v2 — STAGE {stage}: {cfg['label']}")
    print(f"  Epochs: {cfg['epochs']} | LR: {cfg['lr']} | Max Chain: {cfg['max_n']}")
    print(f"{'=' * 70}\n")

    # Dynamic batch size to prevent CUDA OOM on longer sequences
    if stage == 1 or stage == 2:
        stage_batch_size = 128
    elif stage == 3:
        stage_batch_size = 32
    else:
        stage_batch_size = 16
        
    # Load data for this stage
    train_loader, val_loader, test_loader, _ = create_pointer_dataloaders(
        DATA_PATH, batch_size=stage_batch_size, max_chain_len=cfg['max_n']
    )

    if len(train_loader) == 0:
        print(f"  No training data for stage {stage}. Skipping.")
        return model

    # Load previous stage weights if available
    if resume_path and os.path.exists(resume_path):
        print(f"  Loading weights from {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device,
                                         weights_only=True))

    # Optimizer & scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg['lr'],
        steps_per_epoch=len(train_loader),
        epochs=cfg['epochs'],
        pct_start=0.15
    )

    checkpoint_path = os.path.join(MODEL_DIR, f'{MODEL_PREFIX}_checkpoint.pth')
    start_epoch = 0
    
    # Load epoch-level checkpoint if it exists and matches current stage
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if ckpt.get('stage') == stage:
                print(f"  Detected checkpoint for Stage {stage}, Epoch {ckpt['epoch']}. Resuming...")
                model.load_state_dict(ckpt['model_state'])
                optimizer.load_state_dict(ckpt['optimizer_state'])
                scheduler.load_state_dict(ckpt['scheduler_state'])
                start_epoch = ckpt['epoch']
        except Exception as e:
            print(f"  Could not load checkpoint: {e}. Starting stage from beginning.")

    criterion = PointerNetLoss(split_weight=0.9, cost_weight=0.1)

    best_val_loss = float('inf')
    no_improve = 0
    save_path = os.path.join(MODEL_DIR, f'{MODEL_PREFIX}_stage{stage}.pth')
    best_path = os.path.join(MODEL_DIR, f'{MODEL_PREFIX}_best.pth')

    for epoch in range(start_epoch, cfg['epochs']):
        t0 = time.time()

        # Train
        train_loss, train_s, train_c, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        # Validate
        val_loss, val_s, val_c, val_acc = validate(
            model, val_loader, criterion, device
        )

        # Save epoch checkpoint for resuming
        torch.save({
            'stage': stage,
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        }, checkpoint_path)

        elapsed = time.time() - t0

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{cfg['epochs']} | "
                  f"Train Loss: {train_loss:.4f} (S:{train_s:.4f} C:{train_c:.4f}) "
                  f"Acc: {train_acc:.1%} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1%} | "
                  f"{elapsed:.1f}s")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Load best weights from this stage
    model.load_state_dict(torch.load(save_path, map_location=device,
                                     weights_only=True))

    # Evaluate cost accuracy on test set
    print(f"\n  --- Stage {stage} Test Evaluation ---")
    mape, exact_rate, split_acc, failure_rate = evaluate_cost_accuracy(
        model, test_loader, device
    )
    print(f"  Split Accuracy:       {split_acc:.2f}%")
    print(f"  Cost MAPE:            {mape:.4f}%")
    print(f"  Exact Match (<0.1%):  {exact_rate:.1f}%")
    print(f"  Failure Rate (>1%):   {failure_rate:.1f}%")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train Transformer v2 MCM Split Network')
    parser.add_argument('--stage', type=int, default=0,
                        help='Run only this stage (1-4). 0 = all stages.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # 1. Setup
    device = setup_device()
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. Check data exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found: {DATA_PATH}")
        print("  Run: python data/generate_data_v3.py")
        return

    # 3. Create model
    model = TransformerMCMSplitNet(
        input_dim=8,
        d_model=128,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: TransformerMCMSplitNet")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable:,}")

    # Compare with Pointer Network
    from src.models.pointer_mcm import PointerMCMNet
    ptr_model = PointerMCMNet(input_dim=8, d_model=128, nhead=8, num_layers=6)
    ptr_params = sum(p.numel() for p in ptr_model.parameters())
    print(f"\n  (Pointer Network has {ptr_params:,} parameters for reference)")
    del ptr_model

    # 4. Curriculum Training
    resume_path = args.resume

    if args.stage == 0:
        # Run all stages
        for stage in range(1, 5):
            stage_path = os.path.join(MODEL_DIR, f'{MODEL_PREFIX}_stage{stage}.pth')
            if os.path.exists(stage_path) and args.resume is None:
                print(f"\n✅ Stage {stage} checkpoint found. Skipping to next stage.")
                resume_path = stage_path
                continue
                
            model = train_stage(model, stage, device, resume_path)
            resume_path = stage_path
    else:
        # Run specific stage
        if args.stage > 1 and resume_path is None:
            prev = os.path.join(MODEL_DIR, f'{MODEL_PREFIX}_stage{args.stage - 1}.pth')
            if os.path.exists(prev):
                resume_path = prev
        model = train_stage(model, args.stage, device, resume_path)

    # 5. Final comprehensive evaluation
    print(f"\n{'=' * 70}")
    print(f"  TRANSFORMER v2 — FINAL EVALUATION (Full Test Set)")
    print(f"{'=' * 70}\n")

    _, _, test_loader, _ = create_pointer_dataloaders(
        DATA_PATH, batch_size=BATCH_SIZE, max_chain_len=None
    )

    best_path = os.path.join(MODEL_DIR, f'{MODEL_PREFIX}_best.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=True))

    mape, exact_rate, split_acc, failure_rate = evaluate_cost_accuracy(
        model, test_loader, device, max_samples=5000
    )

    print(f"  Overall Split Accuracy:       {split_acc:.2f}%")
    print(f"  Overall Cost MAPE:            {mape:.4f}%")
    print(f"  Exact Match Rate (<0.1%):     {exact_rate:.1f}%")
    print(f"  Failure Rate (>1% error):     {failure_rate:.1f}%")
    print()

    if mape < 1.0:
        print("  ✅ SUCCESS: Target MAPE of <1% ACHIEVED!")
    elif mape < 5.0:
        print(f"  ✅ GOOD: MAPE is {mape:.2f}% -- significant improvement!")
    else:
        print(f"  ⚠️ RESULT: MAPE is {mape:.2f}%. Consider more training or architecture tuning.")

    print(f"\n  Best model saved to: {best_path}")

    # 6. Quick comparison with Pointer Network results (if available)
    ptr_best = os.path.join(MODEL_DIR, 'pointer_best.pth')
    if os.path.exists(ptr_best):
        print(f"\n{'=' * 70}")
        print(f"  HEAD-TO-HEAD COMPARISON")
        print(f"{'=' * 70}")
        print(f"  Pointer Network:    0.0892% MAPE | 95.63% Exact | 98.20% Split Acc")
        print(f"  Transformer v2:     {mape:.4f}% MAPE | {exact_rate:.2f}% Exact | {split_acc:.2f}% Split Acc")

        if mape < 0.0892:
            print(f"\n  🏆 TRANSFORMER v2 WINS!")
        elif mape < 1.0:
            print(f"\n  📊 Both models are excellent. Pointer Network leads by {mape - 0.0892:.4f}% MAPE.")
        else:
            print(f"\n  📊 Pointer Network leads. Transformer v2 gap: {mape - 0.0892:.2f}% MAPE.")


if __name__ == '__main__':
    main()

"""
GNN Training Script for Matrix Chain Multiplication.
=====================================================
Complete training pipeline with:
  - 4-stage curriculum learning (short → long chains)
  - Checkpoint save/resume support
  - Early stopping with patience
  - Gradient clipping
  - Cosine annealing with warm restarts
  - Automatic low-memory mode for large stages (16GB safe)
  - Comprehensive logging

Usage:
    # Train from scratch
    python -m src.training.train_gnn

    # Resume from checkpoint
    python -m src.training.train_gnn --resume models/gnn_checkpoint.pth

    # Train specific curriculum stage
    python -m src.training.train_gnn --stage 3

    # Use workers for speed (safe in low-mem mode)
    python -m src.training.train_gnn --resume models/gnn_checkpoint.pth --stage 3 --workers 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import time
import json
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.gnn_mcm import GraphMCMNet, reconstruct_parenthesization
from src.models.pointer_mcm import compute_cost_from_splits
from src.data.gnn_loader import create_gnn_dataloaders


# ─── Loss Function ───────────────────────────────────────────────────

class GNNSplitLoss(nn.Module):
    """
    Combined loss for GNN split prediction:
      - Primary (90%): Cross-entropy over split predictions, weighted by length
      - Auxiliary (10%): LogCosh on cost predictions
    """

    def __init__(self, split_weight=0.9, cost_weight=0.1):
        super().__init__()
        self.split_weight = split_weight
        self.cost_weight = cost_weight

    def forward(self, split_logits, split_valid, split_targets,
                actual_lengths, aux_cost, cost_targets):
        """
        Returns:
            total_loss, split_loss_value, cost_loss_value, split_accuracy
        """
        total_loss = 0.0
        total_weight = 0.0
        total_correct = 0
        total_count = 0

        for L, logits in split_logits.items():
            valid = split_valid[L]                           # (batch, num_sub)
            targets = split_targets[L]                        # (batch, num_sub)
            batch_size, num_sub, num_cand = logits.shape

            # Flatten valid entries
            valid_flat = valid.reshape(-1)
            logits_flat = logits.reshape(-1, num_cand)
            targets_flat = targets.reshape(-1)

            valid_indices = valid_flat.nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                continue

            valid_logits = logits_flat[valid_indices]
            valid_targets = targets_flat[valid_indices].clamp(0, num_cand - 1)

            # Cross-entropy weighted by sub-chain length
            ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction='mean')
            w = float(L)
            total_loss += w * ce_loss * len(valid_indices)
            total_weight += w * len(valid_indices)

            # Accuracy
            preds = valid_logits.argmax(dim=-1)
            total_correct += (preds == valid_targets).sum().item()
            total_count += len(valid_indices)

        if total_weight == 0:
            s_loss = torch.tensor(0.0, device=aux_cost.device, requires_grad=True)
            s_acc = 0.0
        else:
            s_loss = total_loss / total_weight
            s_acc = total_correct / max(total_count, 1)

        # Cost loss (LogCosh)
        diff = aux_cost - cost_targets
        c_loss = torch.mean(torch.log(torch.cosh(diff + 1e-12)))

        total = self.split_weight * s_loss + self.cost_weight * c_loss

        return total, s_loss.item() if torch.is_tensor(s_loss) else s_loss, c_loss.item(), s_acc


# ─── Training Functions ──────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler, grad_clip=1.0):
    """Train for one epoch. Scaler is passed in (not recreated each epoch)."""
    model.train()
    total_loss = 0
    total_split_loss = 0
    total_cost_loss = 0
    total_acc = 0
    num_batches = 0

    for batch_idx, (node_feat, edge_idx, batch_info, split_targets, cost_targets, _) in enumerate(loader):
        # Move to device
        node_feat = node_feat.to(device, non_blocking=True)
        edge_idx = edge_idx.to(device, non_blocking=True)
        cost_targets = cost_targets.to(device, non_blocking=True)
        batch_info['actual_lengths'] = batch_info['actual_lengths'].to(device, non_blocking=True)
        batch_info['root_indices'] = batch_info['root_indices'].to(device, non_blocking=True)

        # Move split indices to device
        for L in list(batch_info['split_parent_idx'].keys()):
            batch_info['split_parent_idx'][L] = batch_info['split_parent_idx'][L].to(device, non_blocking=True)
            batch_info['split_left_idx'][L] = batch_info['split_left_idx'][L].to(device, non_blocking=True)
            batch_info['split_right_idx'][L] = batch_info['split_right_idx'][L].to(device, non_blocking=True)
            batch_info['split_valid'][L] = batch_info['split_valid'][L].to(device, non_blocking=True)
            split_targets[L] = split_targets[L].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward with autocast for massive speedup on RTX 40-series
        with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            split_logits, split_valid, aux_cost = model(node_feat, edge_idx, batch_info)
            loss, s_loss, c_loss, acc = criterion(
                split_logits, split_valid, split_targets,
                batch_info['actual_lengths'], aux_cost, cost_targets
            )

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_split_loss += s_loss
        total_cost_loss += c_loss
        total_acc += acc
        num_batches += 1

    n = max(num_batches, 1)
    return total_loss / n, total_split_loss / n, total_cost_loss / n, total_acc / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate and compute metrics."""
    model.eval()
    total_loss = 0
    total_split_loss = 0
    total_cost_loss = 0
    total_acc = 0
    num_batches = 0

    # Also compute cost-based metrics on a subset
    cost_errors = []
    valid_count = 0
    total_samples = 0

    for node_feat, edge_idx, batch_info, split_targets, cost_targets, raw_dims in loader:
        node_feat = node_feat.to(device, non_blocking=True)
        edge_idx = edge_idx.to(device, non_blocking=True)
        cost_targets = cost_targets.to(device, non_blocking=True)
        batch_info['actual_lengths'] = batch_info['actual_lengths'].to(device, non_blocking=True)
        batch_info['root_indices'] = batch_info['root_indices'].to(device, non_blocking=True)

        for L in list(batch_info['split_parent_idx'].keys()):
            batch_info['split_parent_idx'][L] = batch_info['split_parent_idx'][L].to(device, non_blocking=True)
            batch_info['split_left_idx'][L] = batch_info['split_left_idx'][L].to(device, non_blocking=True)
            batch_info['split_right_idx'][L] = batch_info['split_right_idx'][L].to(device, non_blocking=True)
            batch_info['split_valid'][L] = batch_info['split_valid'][L].to(device, non_blocking=True)
            split_targets[L] = split_targets[L].to(device, non_blocking=True)

        # Loss
        split_logits, split_valid, aux_cost = model(node_feat, edge_idx, batch_info)
        loss, s_loss, c_loss, acc = criterion(
            split_logits, split_valid, split_targets,
            batch_info['actual_lengths'], aux_cost, cost_targets
        )

        total_loss += loss.item()
        total_split_loss += s_loss
        total_cost_loss += c_loss
        total_acc += acc
        num_batches += 1

        # Compute actual cost from predicted splits (sample a few batches)
        if num_batches <= 5:
            pred_splits, _ = model.predict(node_feat, edge_idx, batch_info)
            for b in range(len(raw_dims)):
                dims = raw_dims[b]
                true_cost = np.expm1(cost_targets[b, 0].cpu().item())
                try:
                    pred_cost = compute_cost_from_splits(dims, pred_splits[b])
                    error = abs(pred_cost - true_cost) / max(true_cost, 1) * 100
                    cost_errors.append(error)
                    if pred_cost >= true_cost - 1:
                        valid_count += 1
                except Exception:
                    pass
                total_samples += 1

    n = max(num_batches, 1)
    metrics = {
        'loss': total_loss / n,
        'split_loss': total_split_loss / n,
        'cost_loss': total_cost_loss / n,
        'split_accuracy': total_acc / n,
    }

    if cost_errors:
        metrics['cost_mape'] = np.mean(cost_errors)
        metrics['validity_rate'] = valid_count / max(total_samples, 1) * 100

    return metrics


# ─── Checkpoint Functions ─────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, stage, best_val_acc,
                    history, config, path):
    """Save a training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'stage': stage,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_accuracy': best_val_acc,
        'history': history,
        'config': config,
        'timestamp': datetime.now().isoformat(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint


# ─── Curriculum Stages ────────────────────────────────────────────────

CURRICULUM_STAGES = [
    {'stage': 1, 'max_n': 10,   'epochs': 30, 'lr': 1e-3, 'batch_size': 128},
    {'stage': 2, 'max_n': 20,   'epochs': 30, 'lr': 5e-4, 'batch_size': 96},
    {'stage': 3, 'max_n': 35,   'epochs': 5,  'lr': 2e-4, 'batch_size': 48},
    {'stage': 4, 'max_n': None, 'epochs': 8,  'lr': 1e-4, 'batch_size': 32},
]


# ─── Main Training Loop ──────────────────────────────────────────────

def train_stage(model, stage_config, data_path, device, model_config,
                history, start_epoch=0, best_val_acc=0.0, patience=15, num_workers=0):
    """
    Train one curriculum stage.

    Args:
        model: GraphMCMNet instance
        stage_config: dict with stage parameters
        data_path: path to dataset
        device: torch device
        model_config: model config dict (for checkpoint)
        history: training history dict
        start_epoch: epoch to resume from (0 for fresh)
        best_val_acc: best validation accuracy so far
        patience: early stopping patience
        num_workers: dataloader workers

    Returns:
        best_val_acc, history
    """
    stage = stage_config['stage']
    max_n = stage_config['max_n']
    epochs = stage_config['epochs']
    lr = stage_config['lr']
    batch_size = stage_config['batch_size']

    # Automatically use low_mem for stages 3 and 4 to prevent OOM on 16GB systems
    use_low_mem = stage >= 3

    print(f"\n{'='*70}")
    print(f"  CURRICULUM STAGE {stage}: n <= {max_n or 'ALL'}")
    print(f"  Epochs: {epochs} | LR: {lr} | Batch: {batch_size}")
    if use_low_mem:
        print(f"  Mode: LOW MEMORY (on-the-fly graphs, workers={num_workers})")
    else:
        print(f"  Mode: CACHED (precomputed graphs)")
    print(f"{'='*70}")

    # Create data loaders for this stage
    train_loader, val_loader, _, _ = create_gnn_dataloaders(
        data_path, batch_size=batch_size, max_chain_len=max_n,
        num_workers=num_workers, low_mem=use_low_mem
    )

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4,
                            betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    criterion = GNNSplitLoss()

    # GradScaler — created ONCE per stage (not per epoch)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    # Early stopping
    no_improve = 0
    stage_best_acc = best_val_acc

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_s, train_c, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        val_acc = val_metrics['split_accuracy']

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        elapsed = time.time() - epoch_start
        cost_info = ""
        if 'cost_mape' in val_metrics:
            cost_info = f" | MAPE: {val_metrics['cost_mape']:.2f}%"
            cost_info += f" | Valid: {val_metrics.get('validity_rate', 0):.1f}%"

        print(f"  S{stage} E{epoch+1:>3}/{epochs} | "
              f"Train: L={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val: L={val_metrics['loss']:.4f} Acc={val_acc:.4f}"
              f"{cost_info} | "
              f"LR={current_lr:.2e} | {elapsed:.1f}s")

        # Track history
        history.setdefault('stages', {}).setdefault(str(stage), {
            'train_losses': [], 'val_losses': [], 'val_accuracies': [],
            'learning_rates': []
        })
        stage_hist = history['stages'][str(stage)]
        stage_hist['train_losses'].append(train_loss)
        stage_hist['val_losses'].append(val_metrics['loss'])
        stage_hist['val_accuracies'].append(val_acc)
        stage_hist['learning_rates'].append(current_lr)

        # Save checkpoint every epoch
        save_checkpoint(
            model, optimizer, scheduler, epoch, stage, stage_best_acc,
            history, model_config, 'models/gnn_checkpoint.pth'
        )

        # Best model?
        if val_acc > stage_best_acc:
            stage_best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), 'models/gnn_best.pth')
            print(f"    ★ New best accuracy: {val_acc:.4f}")
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= patience:
            print(f"    ⏹ Early stopping after {patience} epochs without improvement")
            break

    return stage_best_acc, history


def main():
    parser = argparse.ArgumentParser(description='Train GNN for MCM')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--stage', type=int, default=None,
                        help='Train only a specific curriculum stage (1-4)')
    parser.add_argument('--data', type=str, default='data/mcm_120000.json',
                        help='Path to dataset')
    parser.add_argument('--d-model', type=int, default=128,
                        help='Model hidden dimension')
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of GNN message passing layers')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of dataloader workers (safe to use with low-mem)')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 GNN MCM Training")
    print(f"   Device: {device}")

    # Model config
    model_config = {
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'max_n': 50,
    }

    # Create model
    model = GraphMCMNet(
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        max_n=model_config['max_n'],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {param_count:,}")

    os.makedirs('models', exist_ok=True)

    # Resume or fresh start
    history = {}
    start_stage = 1
    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        print(f"\n📂 Resuming from {args.resume}")
        checkpoint = load_checkpoint(args.resume, model)
        start_stage = checkpoint.get('stage', 1)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('best_val_accuracy', 0.0)
        history = checkpoint.get('history', {})
        print(f"   Stage: {start_stage} | Epoch: {start_epoch} | Best Acc: {best_val_acc:.4f}")

    # Determine which stages to run
    if args.stage:
        stages_to_run = [s for s in CURRICULUM_STAGES if s['stage'] == args.stage]
    else:
        stages_to_run = [s for s in CURRICULUM_STAGES if s['stage'] >= start_stage]

    # Train
    for stage_config in stages_to_run:
        epoch_start = start_epoch if stage_config['stage'] == start_stage else 0
        best_val_acc, history = train_stage(
            model, stage_config, args.data, device, model_config,
            history, start_epoch=epoch_start, best_val_acc=best_val_acc,
            patience=args.patience, num_workers=args.workers
        )

    # Final evaluation
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print(f"  Best model saved to: models/gnn_best.pth")
    print(f"  Checkpoint saved to: models/gnn_checkpoint.pth")
    print(f"{'='*70}")

    # Quick test evaluation
    print("\n📊 Running final test evaluation...")
    model.load_state_dict(torch.load('models/gnn_best.pth', map_location=device,
                                      weights_only=True))
    _, _, test_loader, _ = create_gnn_dataloaders(args.data, batch_size=32, low_mem=True)
    criterion = GNNSplitLoss()
    test_metrics = validate(model, test_loader, criterion, device)

    print(f"\n  TEST RESULTS:")
    print(f"  Split Accuracy: {test_metrics['split_accuracy']:.4f}")
    if 'cost_mape' in test_metrics:
        print(f"  Cost MAPE:      {test_metrics['cost_mape']:.4f}%")
        print(f"  Validity Rate:  {test_metrics.get('validity_rate', 0):.1f}%")

    # Show a few example parenthesizations
    print("\n📝 Example Predictions:")
    model.eval()
    for node_feat, edge_idx, batch_info, _, cost_targets, raw_dims in test_loader:
        node_feat = node_feat.to(device)
        edge_idx = edge_idx.to(device)
        batch_info['actual_lengths'] = batch_info['actual_lengths'].to(device)
        batch_info['root_indices'] = batch_info['root_indices'].to(device)
        for L in batch_info['split_parent_idx']:
            batch_info['split_parent_idx'][L] = batch_info['split_parent_idx'][L].to(device)
            batch_info['split_left_idx'][L] = batch_info['split_left_idx'][L].to(device)
            batch_info['split_right_idx'][L] = batch_info['split_right_idx'][L].to(device)
            batch_info['split_valid'][L] = batch_info['split_valid'][L].to(device)

        pred_splits, _ = model.predict(node_feat, edge_idx, batch_info)

        for b in range(min(3, len(raw_dims))):
            dims = raw_dims[b]
            n = len(dims) - 1
            true_cost = np.expm1(cost_targets[b, 0].item())
            pred_cost = compute_cost_from_splits(dims, pred_splits[b])
            paren = reconstruct_parenthesization(pred_splits[b], n)

            print(f"\n  Chain (n={n}): {dims[:6]}{'...' if n > 5 else ''}")
            print(f"  DP Cost:   {true_cost:,.0f}")
            print(f"  GNN Cost:  {pred_cost:,.0f}")
            print(f"  Error:     {abs(pred_cost - true_cost) / max(true_cost, 1) * 100:.4f}%")
            print(f"  Parens:    {paren[:80]}{'...' if len(paren) > 80 else ''}")
        break  # Only first batch


if __name__ == '__main__':
    main()

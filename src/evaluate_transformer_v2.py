"""
Evaluation Script for Transformer v2 MCM Split Network.
========================================================
Runs comprehensive evaluation on the full test set and compares
head-to-head with the Pointer Network.

Usage:
    python -m src.evaluate_transformer_v2
    python -m src.evaluate_transformer_v2 --model models/transformer_v2_stage3.pth
"""

import torch
import numpy as np
import os
import sys
import argparse
import time

from src.models.transformer_split import TransformerMCMSplitNet
from src.models.pointer_mcm import PointerMCMNet, compute_cost_from_splits
from src.data.pointer_loader import create_pointer_dataloaders


def evaluate_model(model, loader, device, model_name, max_samples=None):
    """
    Evaluate a split-prediction model on the test set.

    Returns dict with all metrics.
    """
    model.eval()
    all_pred_costs = []
    all_true_costs = []
    all_split_accs = []
    all_chain_lengths = []
    count = 0

    print(f"\n  Evaluating {model_name}...")
    t0 = time.time()

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

                pred_cost = compute_cost_from_splits(dims, predicted_splits[b])
                all_pred_costs.append(pred_cost)
                all_true_costs.append(true_cost)
                all_chain_lengths.append(n)

                # Split accuracy
                correct = 0
                total = 0
                for i in range(1, n + 1):
                    for j in range(i + 2, n + 1):
                        true_k = split_tgt[b, i - 1, j - 1].item() + i
                        pred_k = predicted_splits[b].get((i, j), -1)
                        if pred_k == true_k:
                            correct += 1
                        total += 1
                if total > 0:
                    all_split_accs.append(correct / total)

                count += 1
                if max_samples and count >= max_samples:
                    break

                if count % 2000 == 0:
                    elapsed = time.time() - t0
                    print(f"    [{count:>6,}] {elapsed:.1f}s elapsed")

            if max_samples and count >= max_samples:
                break

    pred_arr = np.array(all_pred_costs, dtype=np.float64)
    true_arr = np.array(all_true_costs, dtype=np.float64)
    errors = np.abs(true_arr - pred_arr) / (true_arr + 1e-8)

    results = {
        'model_name': model_name,
        'samples': count,
        'mape': np.mean(errors) * 100,
        'median_ape': np.median(errors) * 100,
        'exact_match_rate': np.mean(errors < 0.001) * 100,
        'near_match_rate': np.mean(errors < 0.01) * 100,
        'failure_rate': np.mean(errors > 0.01) * 100,
        'split_accuracy': np.mean(all_split_accs) * 100 if all_split_accs else 0,
        'max_error': np.max(errors) * 100,
        'p95_error': np.percentile(errors, 95) * 100,
        'p99_error': np.percentile(errors, 99) * 100,
        'eval_time': time.time() - t0,
    }

    # Per chain-length analysis
    lengths = np.array(all_chain_lengths)
    for label, lo, hi in [('short', 3, 10), ('medium', 11, 25), ('long', 26, 40), ('very_long', 41, 50)]:
        mask = (lengths >= lo) & (lengths <= hi)
        if mask.sum() > 0:
            results[f'mape_{label}'] = np.mean(errors[mask]) * 100
            results[f'count_{label}'] = int(mask.sum())

    return results


def print_results(results):
    """Pretty-print evaluation results."""
    name = results['model_name']
    print(f"\n{'=' * 60}")
    print(f"  {name} — EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Samples Evaluated:     {results['samples']:,}")
    print(f"  Evaluation Time:       {results['eval_time']:.1f}s")
    print()
    print(f"  ┌─────────────────────────┬────────────┐")
    print(f"  │ Metric                  │ Value      │")
    print(f"  ├─────────────────────────┼────────────┤")
    print(f"  │ Cost MAPE (mean)        │ {results['mape']:>8.4f}%  │")
    print(f"  │ Cost APE (median)       │ {results['median_ape']:>8.4f}%  │")
    print(f"  │ Split Accuracy          │ {results['split_accuracy']:>8.2f}%  │")
    print(f"  │ Exact Match (<0.1%)     │ {results['exact_match_rate']:>8.2f}%  │")
    print(f"  │ Near Match (<1%)        │ {results['near_match_rate']:>8.2f}%  │")
    print(f"  │ Failure Rate (>1%)      │ {results['failure_rate']:>8.2f}%  │")
    print(f"  │ P95 Error               │ {results['p95_error']:>8.4f}%  │")
    print(f"  │ P99 Error               │ {results['p99_error']:>8.4f}%  │")
    print(f"  │ Max Error               │ {results['max_error']:>8.2f}%  │")
    print(f"  └─────────────────────────┴────────────┘")

    # Per chain-length breakdown
    print(f"\n  Chain Length Breakdown:")
    for label in ['short', 'medium', 'long', 'very_long']:
        key_mape = f'mape_{label}'
        key_count = f'count_{label}'
        if key_mape in results:
            print(f"    {label:>10s} (n={{'short':'3-10','medium':'11-25','long':'26-40','very_long':'41-50'}[label]}): "
                  f"{results[key_mape]:.4f}% MAPE ({results[key_count]:,} samples)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Transformer v2')
    parser.add_argument('--model', type=str, default='models/transformer_v2_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--samples', type=int, default=None,
                        help='Max samples to evaluate (None = all)')
    parser.add_argument('--compare', action='store_true', default=True,
                        help='Also evaluate Pointer Network for comparison')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_path = 'data/mcm_120000.json'
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found: {data_path}")
        return

    # Load test data (same splits as training)
    print("Loading test dataset...")
    _, _, test_loader, _ = create_pointer_dataloaders(
        data_path, batch_size=64, max_chain_len=None
    )

    # ─── Evaluate Transformer v2 ────────────────────────────────────────

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        return

    transformer = TransformerMCMSplitNet(
        input_dim=8, d_model=128, nhead=8,
        num_encoder_layers=6, num_decoder_layers=4,
        dim_feedforward=512, dropout=0.1,
    ).to(device)

    transformer.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    t_results = evaluate_model(transformer, test_loader, device, "Transformer v2", args.samples)
    print_results(t_results)

    # ─── Compare with Pointer Network ───────────────────────────────────

    ptr_path = 'models/pointer_best.pth'
    if args.compare and os.path.exists(ptr_path):
        pointer = PointerMCMNet(
            input_dim=8, d_model=128, nhead=8,
            num_layers=6, dim_feedforward=512, dropout=0.1,
        ).to(device)

        pointer.load_state_dict(torch.load(ptr_path, map_location=device, weights_only=True))
        p_results = evaluate_model(pointer, test_loader, device, "Pointer Network", args.samples)
        print_results(p_results)

        # ─── Head-to-Head Comparison ────────────────────────────────────

        print(f"\n{'=' * 60}")
        print(f"  HEAD-TO-HEAD COMPARISON")
        print(f"{'=' * 60}")
        print(f"  {'Metric':<25} | {'Pointer':>12} | {'Transformer v2':>14} | {'Winner':>10}")
        print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*14}-+-{'-'*10}")

        comparisons = [
            ('Cost MAPE', 'mape', 'lower'),
            ('Split Accuracy', 'split_accuracy', 'higher'),
            ('Exact Match Rate', 'exact_match_rate', 'higher'),
            ('Failure Rate', 'failure_rate', 'lower'),
            ('P99 Error', 'p99_error', 'lower'),
        ]

        for label, key, direction in comparisons:
            p_val = p_results[key]
            t_val = t_results[key]

            if direction == 'lower':
                winner = "Pointer" if p_val < t_val else "Trans v2" if t_val < p_val else "Tie"
            else:
                winner = "Pointer" if p_val > t_val else "Trans v2" if t_val > p_val else "Tie"

            print(f"  {label:<25} | {p_val:>11.4f}% | {t_val:>13.4f}% | {winner:>10}")

        # Parameter comparison
        ptr_params = sum(p.numel() for p in pointer.parameters())
        t_params = sum(p.numel() for p in transformer.parameters())
        print(f"\n  {'Parameters':<25} | {ptr_params:>12,} | {t_params:>14,} |")
        print(f"  {'Eval Time':<25} | {p_results['eval_time']:>11.1f}s | {t_results['eval_time']:>13.1f}s |")

    else:
        print(f"\n  (Pointer Network model not found at {ptr_path}, skipping comparison)")

    print()


if __name__ == '__main__':
    main()

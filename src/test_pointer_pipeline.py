"""
Smoke test for the Pointer Network pipeline.
=============================================
Tests each component with a small synthetic dataset to verify
correctness before running the full training.

Usage: python -m src.test_pointer_pipeline
"""

import torch
import numpy as np
import sys

# Global imports to ensure availability across all test blocks
from data.pointer_features import extract_pointer_features, pad_features
from src.data.pointer_loader import mcm_dp_with_splits
from src.models.pointer_mcm import PointerMCMNet, compute_cost_from_splits
from src.utils.pointer_losses import PointerNetLoss

print("=" * 60)
print("  POINTER NETWORK PIPELINE SMOKE TEST")
print("=" * 60)

errors = []

# ─── Test 1: Pointer Features ────────────────────────────────────────
print("\n[1/6] Testing pointer_features.py...")
try:
    dims = [40, 20, 30, 10, 30]
    feats = extract_pointer_features(dims)
    assert feats.shape == (5, 8), f"Expected (5,8), got {feats.shape}"

    # Position 1 (d=20) should be local_min (20 < 40 and 20 < 30)
    assert feats[1, 6] == 1.0, f"Expected is_local_min=1 at pos 1, got {feats[1, 6]}"

    # Test padding
    padded, mask = pad_features(feats, 51)
    assert padded.shape == (51, 8), f"Expected (51,8), got {padded.shape}"
    assert mask[:5].sum() == 0, "First 5 positions should not be masked"
    assert mask[5:].sum() == 46, "Positions 5-50 should be masked"

    print("  PASS: Features shape and content correct")
except Exception as e:
    errors.append(f"pointer_features: {e}")
    print(f"  FAIL: {e}")

# ─── Test 2: DP Solver with Splits ──────────────────────────────────
print("\n[2/6] Testing DP solver with splits...")
try:
    # Test case 1: [40, 20, 30, 10, 30]
    dims = [40, 20, 30, 10, 30]
    cost, s = mcm_dp_with_splits(dims)
    assert cost == 26000, f"Expected cost 26000, got {cost}"
    # Verify all split entries are within valid range
    n = len(dims) - 1
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            k = s[i][j]
            assert i <= k < j, f"Split s[{i}][{j}]={k} out of range [{i},{j})"

    # Test case 2: [10, 30, 5, 60] — well known answer = 4500
    dims2 = [10, 30, 5, 60]
    cost2, s2 = mcm_dp_with_splits(dims2)
    assert cost2 == 4500, f"Expected cost 4500, got {cost2}"
    assert s2[1][3] == 2, f"Expected s[1][3]=2 for [10,30,5,60], got {s2[1][3]}"

    # Verify cost_from_splits matches DP cost when given optimal splits
    splits_dict = {}
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            splits_dict[(i, j)] = s[i][j]
    recomputed = compute_cost_from_splits(dims, splits_dict)
    assert recomputed == cost, f"Cost from splits {recomputed} != DP cost {cost}"

    print(f"  PASS: cost1={cost}, cost2={cost2}, recomputed={recomputed}")
except Exception as e:
    errors.append(f"DP solver: {e}")
    print(f"  FAIL: {e}")

# ─── Test 3: Model Forward Pass ─────────────────────────────────────
print("\n[3/6] Testing PointerMCMNet forward pass...")
try:
    model = PointerMCMNet(input_dim=8, d_model=64, nhead=4, num_layers=2,
                          dim_feedforward=128, dropout=0.1)
    model.eval()

    batch_size = 4
    seq = torch.randn(batch_size, 51, 8)
    mask = torch.ones(batch_size, 51, dtype=torch.bool)
    actual_n = torch.tensor([4, 6, 3, 10])
    for b in range(batch_size):
        mask[b, :actual_n[b].item() + 1] = False  # unmask real positions

    with torch.no_grad():
        split_logits, split_valid, aux_cost = model(seq, mask, actual_n)

    # Check outputs exist for L=3 and beyond
    assert 3 in split_logits, "Should have logits for L=3"
    assert 2 not in split_logits, "Should NOT have logits for L=2 (trivial)"
    assert aux_cost.shape == (batch_size, 1), f"Aux cost shape wrong: {aux_cost.shape}"

    # Check valid masks
    for L, valid in split_valid.items():
        for b in range(batch_size):
            n = actual_n[b].item()
            expected_valid = max(0, n - L + 1)
            actual_valid = valid[b].sum().item()
            assert actual_valid == expected_valid, \
                f"L={L}, b={b}, n={n}: expected {expected_valid} valid, got {actual_valid}"

    print(f"  PASS: Logits for L=3..{max(split_logits.keys())}, "
          f"aux_cost shape={aux_cost.shape}")
except Exception as e:
    errors.append(f"Model forward: {e}")
    print(f"  FAIL: {e}")

# ─── Test 4: Model Predict + Cost Computation ───────────────────────
print("\n[4/6] Testing predict + cost computation...")
try:
    model = PointerMCMNet(input_dim=8, d_model=64, nhead=4, num_layers=2,
                          dim_feedforward=128, dropout=0.1)

    # Tiny example: 3 matrices, dims = [10, 30, 5, 60]
    test_dims = [10, 30, 5, 60]
    feats = extract_pointer_features(test_dims)
    padded, mask_np = pad_features(feats, 51)

    seq = torch.FloatTensor(padded).unsqueeze(0)
    mask = torch.BoolTensor(mask_np).unsqueeze(0)
    actual_n = torch.tensor([3])

    predicted_splits, aux = model.predict(seq, mask, actual_n)

    # Check we got splits for (1,2), (2,3), and (1,3)
    splits = predicted_splits[0]
    assert (1, 2) in splits, "Missing split (1,2)"
    assert (2, 3) in splits, "Missing split (2,3)"
    assert (1, 3) in splits, "Missing split (1,3)"

    # Compute cost from splits
    pred_cost = compute_cost_from_splits(test_dims, splits)
    assert pred_cost > 0, f"Cost should be positive, got {pred_cost}"

    # True optimal cost for [10,30,5,60]
    true_cost, _ = mcm_dp_with_splits(test_dims)
    print(f"  PASS: pred_cost={pred_cost:,}, true_cost={true_cost:,}, "
          f"splits={(1, 3)}:{splits.get((1, 3))}")
except Exception as e:
    errors.append(f"Predict: {e}")
    print(f"  FAIL: {e}")

# ─── Test 5: Loss Function ──────────────────────────────────────────
print("\n[5/6] Testing loss function...")
try:
    criterion = PointerNetLoss(split_weight=0.9, cost_weight=0.1)

    # Mock data
    split_logits = {
        3: torch.randn(4, 48, 2, requires_grad=True),   # L=3: 2 candidates each
        4: torch.randn(4, 47, 3, requires_grad=True),   # L=4: 3 candidates each
    }
    split_valid = {
        3: torch.zeros(4, 48, dtype=torch.bool),
        4: torch.zeros(4, 47, dtype=torch.bool),
    }
    # Mark some as valid
    split_valid[3][:, :3] = True
    split_valid[4][:, :2] = True

    split_targets = torch.zeros(4, 50, 50, dtype=torch.long)
    actual_n = torch.tensor([5, 5, 5, 5])
    aux_cost = torch.randn(4, 1, requires_grad=True)
    cost_target = torch.randn(4, 1)

    loss, s_loss, c_loss, acc = criterion(
        split_logits, split_valid, split_targets, actual_n, aux_cost, cost_target
    )

    assert loss.requires_grad, "Loss should have grad"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"

    print(f"  PASS: loss={loss.item():.4f}, s_loss={s_loss:.4f}, "
          f"c_loss={c_loss:.4f}, acc={acc:.1%}")
except Exception as e:
    errors.append(f"Loss: {e}")
    print(f"  FAIL: {e}")

# ─── Test 6: Backward Pass ──────────────────────────────────────────
print("\n[6/6] Testing backward pass (gradient flow)...")
try:
    model = PointerMCMNet(input_dim=8, d_model=64, nhead=4, num_layers=2,
                          dim_feedforward=128, dropout=0.1)
    criterion = PointerNetLoss()

    seq = torch.randn(2, 51, 8)
    mask = torch.ones(2, 51, dtype=torch.bool)
    mask[0, :6] = False  # n=5
    mask[1, :8] = False  # n=7
    actual_n = torch.tensor([5, 7])

    # Fake targets
    split_targets = torch.zeros(2, 50, 50, dtype=torch.long)
    cost_target = torch.randn(2, 1)

    # Forward
    split_logits, split_valid, aux_cost = model(seq, mask, actual_n)

    # Loss
    loss, s_loss, c_loss, acc = criterion(
        split_logits, split_valid, split_targets, actual_n, aux_cost, cost_target
    )

    # Backward
    loss.backward()

    # Check gradients exist
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_p = sum(1 for p in model.parameters())
    assert has_grad > 0, "No parameters have gradients!"

    print(f"  PASS: Backward OK. {has_grad}/{total_p} params have gradients. "
          f"Loss={loss.item():.4f}")
except Exception as e:
    errors.append(f"Backward: {e}")
    print(f"  FAIL: {e}")

# ─── Summary ─────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
if errors:
    print(f"  FAILED: {len(errors)} test(s)")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print("  ALL 6 TESTS PASSED!")
    print("\n  Pipeline is ready. Run training with:")
    print("    python -m src.train_pointer")
print("=" * 60)

"""
Dataset and DataLoader for the Pointer Network.
================================================
Loads the original dataset and computes split tables on-the-fly using DP.
Caches per-position features via joblib for speed.
"""

import torch
import numpy as np
import json
import os
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.data.pointer_features import extract_pointer_features, pad_features, FEATURES_PER_POSITION

MAX_LEN = 51   # max n=50 → 51 dimension values
MAX_N = 50     # max number of matrices


def mcm_dp_with_splits(p):
    """
    DP solver returning both minimum cost AND the full split table.
    Returns (cost, s) where s is a list-of-lists with s[i][j] = optimal split k.
    """
    n = len(p) - 1
    m = [[0] * (n + 1) for _ in range(n + 1)]
    s = [[0] * (n + 1) for _ in range(n + 1)]

    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k

    return m[1][n], s


class PointerMCMDataset(Dataset):
    """
    Dataset for Pointer Network MCM training.

    Computes split tables from the DP solver on first access and caches
    the relative split targets as numpy arrays.

    Each sample returns:
        seq_features: (max_len, 8) per-position features
        padding_mask: (max_len,) True for padded positions
        split_targets: (max_n, max_n) target split indices (relative)
        split_mask: (max_n, max_n) True for valid sub-chains
        cost_target: scalar, log1p(cost)
        actual_n: scalar, number of matrices
        raw_dims: list of dimension values (for cost computation)
    """

    def __init__(self, data, precomputed_features, precomputed_splits):
        """
        Args:
            data: list of dicts with 'input', 'output'
            precomputed_features: (N, max_len, 8) numpy array
            precomputed_splits: (N, max_n, max_n) numpy array of relative split targets
        """
        self.data = data
        self.features = precomputed_features
        self.splits = precomputed_splits

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        dims = sample['input']
        cost = sample['output']
        n = len(dims) - 1

        # 1. Per-position features (precomputed)
        seq_features = self.features[idx]
        padding_mask = np.ones(MAX_LEN, dtype=bool)
        padding_mask[:n + 1] = False

        # 2. Split targets (precomputed, already relative)
        split_targets = self.splits[idx]

        # 3. Split mask
        split_mask = np.zeros((MAX_N, MAX_N), dtype=bool)
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                split_mask[i - 1][j - 1] = True

        # 4. Cost target
        cost_target = np.log1p(cost).astype(np.float32)

        return (
            torch.FloatTensor(seq_features),
            torch.BoolTensor(padding_mask),
            torch.LongTensor(split_targets),
            torch.BoolTensor(split_mask),
            torch.FloatTensor([cost_target]),
            torch.LongTensor([n]),
            dims,
        )


def collate_fn(batch):
    """Custom collate that handles variable-length dims lists."""
    seq_features = torch.stack([b[0] for b in batch])
    padding_mask = torch.stack([b[1] for b in batch])
    split_targets = torch.stack([b[2] for b in batch])
    split_mask = torch.stack([b[3] for b in batch])
    cost_target = torch.stack([b[4] for b in batch])
    actual_n = torch.stack([b[5] for b in batch]).squeeze(-1)
    raw_dims = [b[6] for b in batch]

    return seq_features, padding_mask, split_targets, split_mask, cost_target, actual_n, raw_dims


def precompute_all(data, cache_dir='data'):
    """
    Pre-compute features and split targets for all samples.
    Uses joblib caching so this only runs once.

    Returns:
        all_features: (N, MAX_LEN, 8) float32 array
        all_splits: (N, MAX_N, MAX_N) int64 array of relative split targets
    """
    n_samples = len(data)
    feat_cache = os.path.join(cache_dir, f'pointer_features_{n_samples}.joblib')
    split_cache = os.path.join(cache_dir, f'pointer_splits_{n_samples}.joblib')

    # --- Features ---
    if os.path.exists(feat_cache):
        print(f"  Loading cached features: {feat_cache}")
        all_features = joblib.load(feat_cache)
    else:
        print(f"  Computing per-position features for {n_samples:,} samples...")
        all_features = np.zeros((n_samples, MAX_LEN, FEATURES_PER_POSITION), dtype=np.float32)
        for i, sample in enumerate(data):
            feats = extract_pointer_features(sample['input'])
            padded, _ = pad_features(feats, MAX_LEN)
            all_features[i] = padded
            if (i + 1) % 20000 == 0:
                print(f"    [{i + 1:>7,}/{n_samples:,}]")
        print(f"  Saving features cache: {feat_cache}")
        joblib.dump(all_features, feat_cache)

    # --- Split targets ---
    if os.path.exists(split_cache):
        print(f"  Loading cached splits: {split_cache}")
        all_splits = joblib.load(split_cache)
    else:
        print(f"  Computing DP split tables for {n_samples:,} samples...")
        all_splits = np.full((n_samples, MAX_N, MAX_N), -1, dtype=np.int64)
        for i, sample in enumerate(data):
            dims = sample['input']
            n = len(dims) - 1
            cost, s = mcm_dp_with_splits(dims)

            # Verify cost matches stored value
            assert abs(cost - sample['output']) < 1, \
                f"Sample {i}: cost mismatch {cost} vs {sample['output']}"

            # Store relative split indices
            for ii in range(1, n + 1):
                for jj in range(ii + 1, n + 1):
                    all_splits[i, ii - 1, jj - 1] = s[ii][jj] - ii

            if (i + 1) % 10000 == 0:
                print(f"    [{i + 1:>7,}/{n_samples:,}]")

        print(f"  Saving splits cache: {split_cache}")
        joblib.dump(all_splits, split_cache)

    return all_features, all_splits


def create_pointer_dataloaders(data_path, batch_size=128, max_chain_len=None):
    """
    Create train/val/test DataLoaders for the Pointer Network.

    Args:
        data_path: path to original JSON file (e.g. mcm_120000.json)
        batch_size: batch size
        max_chain_len: optional, filter to only chains with n <= this value
                       (used for curriculum learning stages)

    Returns:
        train_loader, val_loader, test_loader, test_indices
    """
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"  Total samples: {len(data):,}")

    # Pre-compute features and splits for ALL data (cached)
    all_features, all_splits = precompute_all(data)

    # Filter by chain length if specified (curriculum learning)
    if max_chain_len is not None:
        keep = [i for i, s in enumerate(data) if len(s['input']) - 1 <= max_chain_len]
        data = [data[i] for i in keep]
        all_features = all_features[keep]
        all_splits = all_splits[keep]
        print(f"  After filtering n <= {max_chain_len}: {len(data):,} samples")

    # Split indices (same split as all other models for fair comparison)
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)

    print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")

    # Create datasets — each gets its own sliced data/features/splits
    train_ds = PointerMCMDataset(
        [data[i] for i in train_idx],
        all_features[train_idx],
        all_splits[train_idx],
    )
    val_ds = PointerMCMDataset(
        [data[i] for i in val_idx],
        all_features[val_idx],
        all_splits[val_idx],
    )
    test_ds = PointerMCMDataset(
        [data[i] for i in test_idx],
        all_features[test_idx],
        all_splits[test_idx],
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, test_idx

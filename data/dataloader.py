"""
ENHANCED DATALOADER v3
======================
131 features: 30 engineered + 51 padded raw dims + 50 pairwise products.

Key additions:
- load_data_split(): Returns identical train/val/test splits for ALL models
- Ensures apples-to-apples comparison across models
"""

import json
import numpy as np
import math
import os
from sklearn.model_selection import train_test_split

MAX_DIMS_LEN = 51  # max n=50 means 51 dimension values
MAX_PAIRS = 50     # max 50 pairwise products

def extract_features(dims):
    """Original 30 engineered features."""
    n = len(dims) - 1
    arr = np.array(dims, dtype=float)

    mn = arr.min()
    mx = arr.max()
    mean = arr.mean()
    std = arr.std() if len(arr) > 1 else 0.0
    med = np.median(arr)
    rng = mx - mn
    cv = std / mean if mean > 0 else 0.0

    log_n = math.log2(n + 1)
    log_mn = math.log10(mn + 1)
    log_mx = math.log10(mx + 1)
    log_mean = math.log10(mean + 1)
    log_std = math.log10(std + 1)

    p25, p75 = np.percentile(arr, [25, 75])
    iqr = p75 - p25

    first3 = math.log10(dims[0] * dims[1] * dims[2] + 1) if n >= 2 else 0
    last3 = math.log10(dims[-3] * dims[-2] * dims[-1] + 1) if n >= 2 else 0
    max_triple = max(dims[i]*dims[i+1]*dims[i+2] for i in range(n-1)) if n >= 2 else dims[0]**3
    log_max_trip = math.log10(max_triple + 1)

    has_bottleneck = 1 if mn <= 3 and mx >= 200 else 0
    has_extreme = 1 if mn == 1 or mx >= 450 else 0
    is_increasing = 1 if list(arr) == sorted(arr) else 0
    is_decreasing = 1 if list(arr) == sorted(arr, reverse=True) else 0
    diversity = len(set(dims)) / len(dims)

    ratios = [dims[i+1]/dims[i] if dims[i] > 0 else 1.0 for i in range(len(dims)-1)]
    ratio_mean = np.mean(ratios)
    ratio_std = np.std(ratios) if len(ratios) > 1 else 0.0
    ratio_max = max(ratios)

    is_small = 1 if n <= 10 else 0
    is_medium = 1 if 10 < n <= 25 else 0
    is_large = 1 if n > 25 else 0

    return [
        n, mn, mx, mean, std, med, rng, cv,
        log_n, log_mn, log_mx, log_mean, log_std,
        p25, p75, iqr,
        first3, last3, log_max_trip,
        has_bottleneck, has_extreme, is_increasing, is_decreasing, diversity,
        ratio_mean, ratio_std, ratio_max,
        is_small, is_medium, is_large
    ]

def extract_features_v2(dims):
    """
    Enhanced feature set: 131 features total.
    - 30 engineered summary features
    - 51 log-scaled padded raw dimensions (captures sequence order)
    - 50 log-scaled pairwise products (captures local cost structure)
    """
    base = extract_features(dims)
    
    # Padded raw dims (log-scaled to normalize)
    log_dims = [math.log10(d + 1) for d in dims]
    padded_dims = log_dims + [0.0] * (MAX_DIMS_LEN - len(log_dims))
    
    # Pairwise products: log10(dims[i] * dims[i+1]) - building blocks of MCM cost
    pairs = [math.log10(dims[i] * dims[i+1] + 1) for i in range(len(dims) - 1)]
    padded_pairs = pairs + [0.0] * (MAX_PAIRS - len(pairs))
    
    return base + padded_dims + padded_pairs

def load_data(version='v2'):
    """
    Load the 50k dataset.
    version='v1': Original 30 features
    version='v2': Enhanced 131 features (default)
    """
    data_path = os.path.join(os.path.dirname(__file__), 'mcm_50000.json')
    with open(data_path) as f:
        data = json.load(f)

    feat_fn = extract_features if version == 'v1' else extract_features_v2

    X = []
    y = []
    raw_dims = []
    for item in data:
        dims = item["input"]
        X.append(feat_fn(dims))
        y.append(item["output"])
        raw_dims.append(dims)

    return np.array(X), np.array(y), raw_dims

def load_data_split(version='v2', test_size=0.15, val_size=0.15):
    """
    Load data and return IDENTICAL train/val/test splits.
    Every model MUST use this to ensure fair comparison.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, 
        dims_train, dims_val, dims_test
    """
    X, y, raw_dims = load_data(version)
    
    # Same random_state everywhere → identical split for all models
    idx = np.arange(len(X))
    idx_train_val, idx_test = train_test_split(idx, test_size=test_size, random_state=42)
    val_frac = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(idx_train_val, test_size=val_frac, random_state=42)
    
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    dims_train = [raw_dims[i] for i in idx_train]
    dims_val = [raw_dims[i] for i in idx_val]
    dims_test = [raw_dims[i] for i in idx_test]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, dims_train, dims_val, dims_test
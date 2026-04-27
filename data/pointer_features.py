"""
Per-Position Feature Extractor for the Pointer Network.
========================================================
8 features per position in the dimension sequence.

These features are designed for the Transformer encoder —
each position carries LOCAL context about its role in the chain.

Feature List:
  0. log_dim:              log1p(d_i)
  1. log_left_product:     log1p(d_{i-1} * d_i)  [0 for i=0]
  2. log_right_product:    log1p(d_i * d_{i+1})   [0 for i=n]
  3. log_triple:           log1p(d_{i-1} * d_i * d_{i+1})  [0 for boundaries]
  4. relative_position:    i / n
  5. relative_magnitude:   d_i / max(dims)
  6. is_local_min:         1 if d_i < neighbors, else 0
  7. is_local_max:         1 if d_i > neighbors, else 0
"""

import math
import numpy as np

FEATURES_PER_POSITION = 8


def extract_pointer_features(dims):
    """
    Extract per-position features for the Pointer Network encoder.

    Args:
        dims: list of n+1 dimension values (integers)

    Returns:
        features: numpy array of shape (n+1, 8)
    """
    n = len(dims) - 1  # number of matrices
    arr = np.array(dims, dtype=float)
    max_dim = arr.max()

    features = np.zeros((n + 1, FEATURES_PER_POSITION), dtype=np.float32)

    for i in range(n + 1):
        # Feature 0: log of dimension value
        features[i, 0] = math.log1p(dims[i])

        # Feature 1: log of left pairwise product
        if i > 0:
            features[i, 1] = math.log1p(dims[i - 1] * dims[i])

        # Feature 2: log of right pairwise product
        if i < n:
            features[i, 2] = math.log1p(dims[i] * dims[i + 1])

        # Feature 3: log of triple product (local multiplication cost)
        if 0 < i < n:
            features[i, 3] = math.log1p(dims[i - 1] * dims[i] * dims[i + 1])

        # Feature 4: relative position in the chain (0 to 1)
        features[i, 4] = i / max(n, 1)

        # Feature 5: relative magnitude (0 to 1)
        features[i, 5] = dims[i] / max(max_dim, 1)

        # Feature 6: is local minimum (bottleneck indicator)
        if 0 < i < n:
            features[i, 6] = 1.0 if dims[i] < dims[i - 1] and dims[i] < dims[i + 1] else 0.0

        # Feature 7: is local maximum
        if 0 < i < n:
            features[i, 7] = 1.0 if dims[i] > dims[i - 1] and dims[i] > dims[i + 1] else 0.0

    return features


def pad_features(features, max_len=51):
    """
    Pad feature array to max_len and create a padding mask.

    Args:
        features: numpy array of shape (actual_len, 8)
        max_len: maximum sequence length (default 51 for n=50)

    Returns:
        padded: numpy array of shape (max_len, 8)
        mask: boolean array of shape (max_len,) — True for padded positions
    """
    actual_len = features.shape[0]
    padded = np.zeros((max_len, FEATURES_PER_POSITION), dtype=np.float32)
    padded[:actual_len] = features

    mask = np.ones(max_len, dtype=bool)
    mask[:actual_len] = False

    return padded, mask


if __name__ == '__main__':
    # Quick test
    test_dims = [40, 20, 30, 10, 30]
    feats = extract_pointer_features(test_dims)
    print(f"Test chain: {test_dims}")
    print(f"Features shape: {feats.shape}  (expected (5, 8))")
    print()

    feature_names = [
        'log_dim', 'log_left_prod', 'log_right_prod', 'log_triple',
        'rel_position', 'rel_magnitude', 'is_local_min', 'is_local_max'
    ]
    for i in range(len(test_dims)):
        print(f"  Position {i} (d={test_dims[i]:3d}): ", end='')
        for j, name in enumerate(feature_names):
            print(f"{name}={feats[i, j]:.3f}  ", end='')
        print()

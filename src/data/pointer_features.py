import math
import numpy as np

FEATURES_PER_POSITION = 8

def extract_pointer_features(dims):
    n = len(dims) - 1
    arr = np.array(dims, dtype=float)
    max_dim = arr.max()
    features = np.zeros((n + 1, FEATURES_PER_POSITION), dtype=np.float32)
    for i in range(n + 1):
        features[i, 0] = math.log1p(dims[i])
        if i > 0: features[i, 1] = math.log1p(dims[i - 1] * dims[i])
        if i < n: features[i, 2] = math.log1p(dims[i] * dims[i + 1])
        if 0 < i < n: features[i, 3] = math.log1p(dims[i - 1] * dims[i] * dims[i + 1])
        features[i, 4] = i / max(n, 1)
        features[i, 5] = dims[i] / max(max_dim, 1)
        if 0 < i < n:
            features[i, 6] = 1.0 if dims[i] < dims[i - 1] and dims[i] < dims[i + 1] else 0.0
            features[i, 7] = 1.0 if dims[i] > dims[i - 1] and dims[i] > dims[i + 1] else 0.0
    return features

def pad_features(features, max_len=51):
    actual_len = features.shape[0]
    padded = np.zeros((max_len, FEATURES_PER_POSITION), dtype=np.float32)
    padded[:actual_len] = features
    mask = np.ones(max_len, dtype=bool)
    mask[:actual_len] = False
    return padded, mask

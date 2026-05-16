import numpy as np
import math
from scipy import stats as scipy_stats
from src.data.generator import (
    greedy_cost_left_to_right, greedy_cost_right_to_left, 
    greedy_cost_min_first, greedy_cost_balanced
)

MAX_DIMS_LEN = 51
MAX_PAIRS = 50

def extract_features_v1(dims):
    n = len(dims) - 1
    arr = np.array(dims, dtype=float)
    mn, mx, mean = arr.min(), arr.max(), arr.mean()
    std = arr.std() if len(arr) > 1 else 0.0
    med = np.median(arr)
    rng, cv = mx - mn, (std / mean if mean > 0 else 0.0)
    log_n, log_mn, log_mx = math.log2(n + 1), math.log1p(mn), math.log1p(mx)
    log_mean, log_std = math.log1p(mean), math.log1p(std)
    p25, p75 = np.percentile(arr, [25, 75])
    iqr = p75 - p25
    first3 = math.log1p(dims[0] * dims[1] * dims[2]) if n >= 2 else 0
    last3 = math.log1p(dims[-3] * dims[-2] * dims[-1]) if n >= 2 else 0
    max_triple = max(dims[i] * dims[i + 1] * dims[i + 2] for i in range(n - 1)) if n >= 2 else dims[0] ** 3
    log_max_trip = math.log1p(max_triple)
    has_bottleneck = 1 if mn <= 3 and mx >= 200 else 0
    has_extreme = 1 if mn == 1 or mx >= 450 else 0
    is_increasing = 1 if list(arr) == sorted(arr) else 0
    is_decreasing = 1 if list(arr) == sorted(arr, reverse=True) else 0
    diversity = len(set(dims)) / len(dims)
    ratios = [dims[i + 1] / dims[i] if dims[i] > 0 else 1.0 for i in range(len(dims) - 1)]
    ratio_mean, ratio_std, ratio_max = np.mean(ratios), (np.std(ratios) if len(ratios) > 1 else 0.0), max(ratios)
    is_small, is_medium, is_large = (1 if n <= 10 else 0), (1 if 10 < n <= 25 else 0), (1 if n > 25 else 0)
    return [n, mn, mx, mean, std, med, rng, cv, log_n, log_mn, log_mx, log_mean, log_std, p25, p75, iqr, first3, last3, log_max_trip, has_bottleneck, has_extreme, is_increasing, is_decreasing, diversity, ratio_mean, ratio_std, ratio_max, is_small, is_medium, is_large]

def extract_position_features(dims):
    arr = np.array(dims, dtype=float)
    n = len(dims) - 1
    sorted_indices = np.argsort(arr)
    min_pos = sorted_indices[0] / max(n, 1)
    min2_pos = sorted_indices[1] / max(n, 1) if len(arr) > 1 else min_pos
    max_pos = sorted_indices[-1] / max(n, 1)
    q = max(len(arr) // 4, 1)
    q1_mean = arr[:q].mean()
    q4_mean = arr[3 * q:].mean() if 3 * q < len(arr) else arr[-1]
    start_log, end_log = math.log1p(dims[0]), math.log1p(dims[-1])
    slope = np.polyfit(np.arange(len(arr)), arr, 1)[0] if len(arr) > 1 else 0.0
    return [min_pos, min2_pos, max_pos, q1_mean, q4_mean, start_log, end_log, slope]

def extract_cost_proxy_features(dims):
    g_lr, g_rl, g_min, g_bal = greedy_cost_left_to_right(dims), greedy_cost_right_to_left(dims), greedy_cost_min_first(dims), greedy_cost_balanced(dims)
    log_lr, log_rl, log_min, log_bal = math.log1p(g_lr), math.log1p(g_rl), math.log1p(g_min), math.log1p(g_bal)
    log_greedy_min_all = math.log1p(min(g_lr, g_rl, g_min, g_bal))
    log_greedy_max_all = math.log1p(max(g_lr, g_rl, g_min, g_bal))
    return [log_lr, log_rl, log_min, log_bal, log_greedy_min_all, log_greedy_max_all]

def extract_features_v4(dims):
    v1 = extract_features_v1(dims)
    pos = extract_position_features(dims)
    proxy = extract_cost_proxy_features(dims)
    # Padded sequence features
    log_dims = [math.log1p(d) for d in dims]
    padded_dims = log_dims + [0.0] * (MAX_DIMS_LEN - len(log_dims))
    pairs = [math.log1p(dims[i] * dims[i + 1]) for i in range(len(dims) - 1)]
    padded_pairs = pairs + [0.0] * (MAX_PAIRS - len(pairs))
    return v1 + pos + proxy + padded_dims + padded_pairs

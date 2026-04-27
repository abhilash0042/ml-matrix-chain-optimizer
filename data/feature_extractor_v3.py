"""
Research-Level Feature Extractor v3
====================================
177 features total:
  - 30 original summary statistics
  - 12 position-aware features
  -  8 cost-proxy features (greedy heuristics) ← KEY INNOVATION
  - 10 triple product statistics
  -  8 spectral/structural features
  -  8 pairwise product statistics
  - 51 padded log-dimensions (sequence order)
  - 50 padded log pairwise products

The greedy cost features are the most important addition:
they give the model a "ballpark" of the answer, so it only
needs to learn a small correction factor (greedy → optimal).
"""

import numpy as np
import math
from scipy import stats as scipy_stats


# ─── Greedy Cost Heuristics ────────────────────────────────────────────
# These are O(n²) approximations to the O(n³) DP solution.
# They are NOT optimal, but they correlate strongly with optimal cost.
# Used as INPUT FEATURES, not as targets.

def greedy_cost_left_to_right(dims):
    """
    Multiply matrices left-to-right sequentially.
    M1*M2, then (M1*M2)*M3, etc.
    """
    if len(dims) <= 2:
        return 0
    total_cost = 0
    current_rows = dims[0]
    current_cols = dims[1]
    for i in range(2, len(dims)):
        next_cols = dims[i]
        total_cost += current_rows * current_cols * next_cols
        current_cols = next_cols
    return total_cost


def greedy_cost_right_to_left(dims):
    """
    Multiply matrices right-to-left sequentially.
    M(n-1)*Mn, then M(n-2)*(M(n-1)*Mn), etc.
    """
    if len(dims) <= 2:
        return 0
    total_cost = 0
    current_rows = dims[-2]
    current_cols = dims[-1]
    for i in range(len(dims) - 3, -1, -1):
        next_rows = dims[i]
        total_cost += next_rows * current_rows * current_cols
        current_rows = next_rows
    return total_cost


def greedy_cost_min_first(dims):
    """
    Greedy: always pick the split point k that minimizes p[k]
    (the shared dimension). This picks the cheapest local multiplication first.
    """
    if len(dims) <= 2:
        return 0

    # Work with a mutable list of dimensions
    d = list(dims)
    total_cost = 0

    while len(d) > 2:
        # Find the position k (1..len-2) with minimum d[k]
        # Multiplying matrices k-1 and k costs d[k-1]*d[k]*d[k+1]
        best_k = 1
        best_cost = d[0] * d[1] * d[2]
        for k in range(1, len(d) - 1):
            c = d[k - 1] * d[k] * d[k + 1]
            if d[k] < d[best_k] or (d[k] == d[best_k] and c < best_cost):
                best_k = k
                best_cost = d[k - 1] * d[k] * d[k + 1]
        total_cost += best_cost
        d.pop(best_k)

    return total_cost


def greedy_cost_balanced(dims):
    """
    Always split the chain in the middle (balanced bisection).
    Recursive with memoization.
    """
    n = len(dims) - 1
    if n <= 0:
        return 0

    memo = {}

    def _solve(i, j):
        if i == j:
            return 0
        if (i, j) in memo:
            return memo[(i, j)]
        k = (i + j) // 2  # always pick middle
        cost = _solve(i, k) + _solve(k + 1, j) + dims[i - 1] * dims[k] * dims[j]
        memo[(i, j)] = cost
        return cost

    return _solve(1, n)


# ─── Feature Extraction Functions ──────────────────────────────────────

MAX_DIMS_LEN = 51   # max n=50 → 51 dimension values
MAX_PAIRS = 50       # max 50 pairwise products


def extract_features_v1(dims):
    """Original 30 engineered features (kept for compatibility)."""
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
    log_mn = math.log1p(mn)
    log_mx = math.log1p(mx)
    log_mean = math.log1p(mean)
    log_std = math.log1p(std)

    p25, p75 = np.percentile(arr, [25, 75])
    iqr = p75 - p25

    first3 = math.log1p(dims[0] * dims[1] * dims[2]) if n >= 2 else 0
    last3 = math.log1p(dims[-3] * dims[-2] * dims[-1]) if n >= 2 else 0
    max_triple = max(dims[i] * dims[i + 1] * dims[i + 2]
                     for i in range(n - 1)) if n >= 2 else dims[0] ** 3
    log_max_trip = math.log1p(max_triple)

    has_bottleneck = 1 if mn <= 3 and mx >= 200 else 0
    has_extreme = 1 if mn == 1 or mx >= 450 else 0
    is_increasing = 1 if list(arr) == sorted(arr) else 0
    is_decreasing = 1 if list(arr) == sorted(arr, reverse=True) else 0
    diversity = len(set(dims)) / len(dims)

    ratios = [dims[i + 1] / dims[i] if dims[i] > 0 else 1.0
              for i in range(len(dims) - 1)]
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


def extract_position_features(dims):
    """12 position-aware features."""
    arr = np.array(dims, dtype=float)
    n = len(dims) - 1

    sorted_indices = np.argsort(arr)

    # Where are the smallest dims? (normalized 0-1)
    min_pos = sorted_indices[0] / max(n, 1)
    min2_pos = sorted_indices[1] / max(n, 1) if len(arr) > 1 else min_pos
    min3_pos = sorted_indices[2] / max(n, 1) if len(arr) > 2 else min2_pos
    max_pos = sorted_indices[-1] / max(n, 1)

    # Quarter-based statistics
    q = max(len(arr) // 4, 1)
    q1_mean = arr[:q].mean()
    q2_mean = arr[q:2 * q].mean() if 2 * q <= len(arr) else arr[q:].mean()
    q3_mean = arr[2 * q:3 * q].mean() if 3 * q <= len(arr) else arr[2 * q:].mean()
    q4_mean = arr[3 * q:].mean() if 3 * q < len(arr) else arr[-1]

    # Start/end characterization
    start_log = math.log1p(dims[0])
    end_log = math.log1p(dims[-1])
    start_end_ratio = dims[0] / max(dims[-1], 1)

    # Trend slope (linear regression slope)
    x = np.arange(len(arr))
    if len(arr) > 1:
        slope = np.polyfit(x, arr, 1)[0]
    else:
        slope = 0.0

    return [
        min_pos, min2_pos, min3_pos, max_pos,
        q1_mean, q2_mean, q3_mean, q4_mean,
        start_log, end_log, start_end_ratio, slope
    ]


def extract_cost_proxy_features(dims):
    """
    8 cost-proxy features using greedy heuristics.
    These are the MOST IMPORTANT new features.
    They give the model a ballpark of the answer.
    """
    g_lr = greedy_cost_left_to_right(dims)
    g_rl = greedy_cost_right_to_left(dims)
    g_min = greedy_cost_min_first(dims)
    g_bal = greedy_cost_balanced(dims)

    log_lr = math.log1p(g_lr)
    log_rl = math.log1p(g_rl)
    log_min = math.log1p(g_min)
    log_bal = math.log1p(g_bal)

    # Asymmetry: how different are L→R vs R→L?
    lr_rl_ratio = g_lr / max(g_rl, 1)

    # Min of all greedy estimates (closest to optimal)
    log_greedy_min_all = math.log1p(min(g_lr, g_rl, g_min, g_bal))

    # Max of all greedy estimates
    log_greedy_max_all = math.log1p(max(g_lr, g_rl, g_min, g_bal))

    # Spread of greedy estimates (low spread = easier problem)
    greedy_spread = log_greedy_max_all - log_greedy_min_all

    return [
        log_lr, log_rl, log_min, log_bal,
        lr_rl_ratio, log_greedy_min_all, log_greedy_max_all, greedy_spread
    ]


def extract_triple_features(dims):
    """10 triple product statistics."""
    n = len(dims) - 1

    if n < 2:
        return [0.0] * 10

    triples = [dims[i] * dims[i + 1] * dims[i + 2] for i in range(n - 1)]
    arr = np.array(triples, dtype=float)
    log_arr = np.log1p(arr)

    log_sum = math.log1p(sum(triples))
    log_mean = np.mean(log_arr)
    log_std = np.std(log_arr) if len(log_arr) > 1 else 0.0
    log_min = np.min(log_arr)
    log_max = np.max(log_arr)
    log_med = np.median(log_arr)
    triple_cv = log_std / max(log_mean, 1e-10)

    if len(arr) >= 3:
        skew = float(scipy_stats.skew(log_arr))
        kurt = float(scipy_stats.kurtosis(log_arr))
    else:
        skew = 0.0
        kurt = 0.0

    # Handle NaN/Inf
    skew = 0.0 if not np.isfinite(skew) else skew
    kurt = 0.0 if not np.isfinite(kurt) else kurt

    return [log_sum, log_mean, log_std, log_min, log_max,
            log_med, triple_cv, skew, kurt,
            float(len(triples))]


def extract_spectral_features(dims):
    """8 spectral/structural features."""
    arr = np.array(dims, dtype=float)
    n = len(arr)

    # Autocorrelation at lag 1 and 2
    if n > 2:
        mean_val = arr.mean()
        var_val = arr.var()
        if var_val > 0:
            ac1 = np.mean((arr[:-1] - mean_val) * (arr[1:] - mean_val)) / var_val
            ac2 = np.mean((arr[:-2] - mean_val) * (arr[2:] - mean_val)) / var_val if n > 3 else 0.0
        else:
            ac1, ac2 = 0.0, 0.0
    else:
        ac1, ac2 = 0.0, 0.0

    # FFT features (captures periodic structure like spiky chains)
    if n >= 4:
        fft_vals = np.abs(np.fft.rfft(arr - arr.mean()))
        fft_vals = fft_vals[1:]  # skip DC component
        if len(fft_vals) > 0 and fft_vals.sum() > 0:
            dominant_freq = np.argmax(fft_vals) / len(fft_vals)
            top3_energy = np.sum(np.sort(fft_vals)[-3:]) if len(fft_vals) >= 3 else fft_vals.sum()
            energy_ratio = top3_energy / fft_vals.sum()
        else:
            dominant_freq, energy_ratio = 0.0, 0.0
    else:
        dominant_freq, energy_ratio = 0.0, 0.0

    # Local extrema counts
    local_min = 0
    local_max = 0
    for i in range(1, n - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            local_min += 1
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            local_max += 1

    # Run lengths (longest consecutive run below/above median)
    med = np.median(arr)
    run_small = 0
    run_large = 0
    cur_small = 0
    cur_large = 0
    for v in arr:
        if v < med:
            cur_small += 1
            cur_large = 0
        else:
            cur_large += 1
            cur_small = 0
        run_small = max(run_small, cur_small)
        run_large = max(run_large, cur_large)

    return [ac1, ac2, dominant_freq, energy_ratio,
            local_min, local_max, run_small, run_large]


def extract_pairwise_stats(dims):
    """8 pairwise product statistics."""
    if len(dims) < 2:
        return [0.0] * 8

    pairs = [dims[i] * dims[i + 1] for i in range(len(dims) - 1)]
    log_pairs = np.log1p(np.array(pairs, dtype=float))

    log_sum = math.log1p(sum(pairs))
    log_mean = np.mean(log_pairs)
    log_std = np.std(log_pairs) if len(log_pairs) > 1 else 0.0
    log_min = np.min(log_pairs)
    log_max = np.max(log_pairs)
    pw_cv = log_std / max(log_mean, 1e-10)

    if len(log_pairs) >= 3:
        skew = float(scipy_stats.skew(log_pairs))
        kurt = float(scipy_stats.kurtosis(log_pairs))
    else:
        skew, kurt = 0.0, 0.0

    skew = 0.0 if not np.isfinite(skew) else skew
    kurt = 0.0 if not np.isfinite(kurt) else kurt

    return [log_sum, log_mean, log_std, log_min, log_max,
            pw_cv, skew, kurt]


def extract_padded_sequence(dims):
    """101 features: 51 padded log-dims + 50 padded pairwise products."""
    # Padded raw dims (log-scaled)
    log_dims = [math.log1p(d) for d in dims]
    padded_dims = log_dims + [0.0] * (MAX_DIMS_LEN - len(log_dims))

    # Pairwise products: log1p(dims[i] * dims[i+1])
    pairs = [math.log1p(dims[i] * dims[i + 1]) for i in range(len(dims) - 1)]
    padded_pairs = pairs + [0.0] * (MAX_PAIRS - len(pairs))

    return padded_dims + padded_pairs


# ─── Main Feature Extraction API ──────────────────────────────────────

def extract_features_v3(dims):
    """
    Research-level feature extraction: 177 features total.

    Feature groups:
      [ 0:30]  Original summary statistics
      [30:42]  Position-aware features
      [42:50]  Cost-proxy features (greedy heuristics)
      [50:60]  Triple product statistics
      [60:68]  Spectral/structural features
      [68:76]  Pairwise product statistics
      [76:177] Padded sequence (51 dims + 50 pairs)
    """
    feats = []

    # Group 1: Original 30 summary features [0:30]
    feats.extend(extract_features_v1(dims))

    # Group 2: Position-aware features [30:42]
    feats.extend(extract_position_features(dims))

    # Group 3: Cost-proxy features [42:50]
    feats.extend(extract_cost_proxy_features(dims))

    # Group 4: Triple product statistics [50:60]
    feats.extend(extract_triple_features(dims))

    # Group 5: Spectral/structural features [60:68]
    feats.extend(extract_spectral_features(dims))

    # Group 6: Pairwise product statistics [68:76]
    feats.extend(extract_pairwise_stats(dims))

    # Group 7: Padded sequence features [76:177]
    feats.extend(extract_padded_sequence(dims))

    return feats


# Feature group names for analysis/debugging
FEATURE_GROUPS = {
    'summary': (0, 30),
    'position': (30, 42),
    'cost_proxy': (42, 50),
    'triple_stats': (50, 60),
    'spectral': (60, 68),
    'pairwise_stats': (68, 76),
    'padded_sequence': (76, 177),
}

TOTAL_FEATURES_V3 = 177


# ─── Quick Test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with a sample chain
    test_dims = [40, 20, 30, 10, 30]
    feats = extract_features_v3(test_dims)
    print(f"Test chain: {test_dims}")
    print(f"Total features: {len(feats)} (expected {TOTAL_FEATURES_V3})")
    print()

    for name, (start, end) in FEATURE_GROUPS.items():
        print(f"  {name:20s} [{start:3d}:{end:3d}] = {feats[start:end][:5]}...")

    print()
    print("Greedy cost estimates:")
    print(f"  Left-to-Right: {greedy_cost_left_to_right(test_dims):,}")
    print(f"  Right-to-Left: {greedy_cost_right_to_left(test_dims):,}")
    print(f"  Min-First:     {greedy_cost_min_first(test_dims):,}")
    print(f"  Balanced:      {greedy_cost_balanced(test_dims):,}")

    # Known optimal for [40,20,30,10,30] = 26,000
    from generate_data_v3 import mcm_dp
    print(f"  Exact DP:      {mcm_dp(test_dims):,}")

"""
Research-Level Feature Extractor v4 (Tree Model Optimized)
==========================================================
213 features total:
  - 177 from v3 (including greedy cost proxies)
  - 36 new features explicitly designed for tree models:
    - Greedy ratios
    - Recursive depth proxies
    - Dimension interaction
    - Sliding window stats
    - Pattern detectors
    - Cost bounds
"""

import numpy as np
import math
from scipy import stats as scipy_stats
import sys
import os

# Import v3 features
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor_v3 import (
    extract_features_v3, greedy_cost_left_to_right, 
    greedy_cost_right_to_left, greedy_cost_min_first, greedy_cost_balanced,
    TOTAL_FEATURES_V3
)


def extract_greedy_ratios(dims):
    """6 features: Ratios between greedy heuristics."""
    g_lr = greedy_cost_left_to_right(dims)
    g_rl = greedy_cost_right_to_left(dims)
    g_min = greedy_cost_min_first(dims)
    g_bal = greedy_cost_balanced(dims)
    
    g_best = min(g_lr, g_rl, g_min, g_bal) + 1e-9
    g_worst = max(g_lr, g_rl, g_min, g_bal) + 1e-9
    
    r_lr = g_lr / g_best
    r_rl = g_rl / g_best
    r_bal = g_bal / g_best
    r_spread = g_worst / g_best
    log_spread = math.log1p(r_spread - 1)
    
    # Consensus: are all heuristics agreeing? (1.0 = yes)
    consensus = 1.0 if r_spread < 1.1 else (1.1 / r_spread)
    
    return [r_lr, r_rl, r_bal, r_spread, log_spread, consensus]


def extract_recursive_depth_proxies(dims):
    """4 features: Proxies for problem scale/complexity."""
    n = len(dims) - 1
    
    if n <= 1:
        return [0.0, 0.0, 0.0, 0.0]
        
    log_n = math.log2(n)
    total_subproblems = n * (n - 1) / 2
    n_cubed = n ** 3
    
    # Approx catalan number log
    # C_n ~ 4^n / (n^{3/2} * sqrt(pi))
    if n > 0:
        log_catalan = n * math.log(4) - 1.5 * math.log(n) - 0.5 * math.log(math.pi)
    else:
        log_catalan = 0.0
        
    return [log_n, total_subproblems, n_cubed, log_catalan]


def extract_dimension_interactions(dims):
    """8 features: How the bottleneck interacts with the chain."""
    n = len(dims) - 1
    if n < 2:
        return [0.0] * 8
        
    arr = np.array(dims, dtype=float)
    min_val = arr.min()
    max_val = arr.max()
    med_val = np.median(arr)
    
    inter_first = math.log1p(min_val * arr[0])
    inter_last = math.log1p(min_val * arr[-1])
    inter_max = math.log1p(min_val * max_val)
    inter_med = math.log1p(min_val * med_val)
    
    ratios = [dims[i+1] / max(1, dims[i]) for i in range(len(dims)-1)]
    log_ratios = np.log1p(np.array(ratios))
    
    max_ratio = np.max(log_ratios)
    min_ratio = np.min(log_ratios)
    
    if len(log_ratios) >= 3:
        skew_ratio = float(scipy_stats.skew(log_ratios))
        kurt_ratio = float(scipy_stats.kurtosis(log_ratios))
    else:
        skew_ratio, kurt_ratio = 0.0, 0.0
        
    skew_ratio = 0.0 if not np.isfinite(skew_ratio) else skew_ratio
    kurt_ratio = 0.0 if not np.isfinite(kurt_ratio) else kurt_ratio
    
    return [inter_first, inter_last, inter_max, inter_med, 
            max_ratio, min_ratio, skew_ratio, kurt_ratio]


def extract_sliding_windows(dims):
    """10 features: Local cost landscape using sliding windows."""
    n = len(dims) - 1
    if n < 2:
        return [0.0] * 10
        
    triples = [dims[i] * dims[i+1] * dims[i+2] for i in range(n-1)]
    log_triples = np.log1p(np.array(triples, dtype=float))
    
    # window-3 max
    w3_max = 0.0
    w3_min = 0.0
    w3_mean = 0.0
    w3_min_pos = 0.0
    
    if len(log_triples) > 0:
        w3_max = np.max(log_triples)
        w3_min = np.min(log_triples)
        w3_mean = np.mean(log_triples)
        w3_min_pos = np.argmin(log_triples) / max(1, len(log_triples)-1)
        
    # window-5 
    quints = [dims[i] * dims[i+1] * dims[i+2] * dims[i+3] * dims[i+4] for i in range(n-3)] if n >= 4 else []
    log_quints = np.log1p(np.array(quints, dtype=float)) if quints else np.array([])
    
    w5_max = 0.0
    w5_min = 0.0
    w5_mean = 0.0
    
    if len(log_quints) > 0:
        w5_max = np.max(log_quints)
        w5_min = np.min(log_quints)
        w5_mean = np.mean(log_quints)
        
    # Diff stats
    diffs = np.diff(log_triples) if len(log_triples) > 1 else np.array([0.0])
    max_jump = np.max(np.abs(diffs)) if len(diffs) > 0 else 0.0
    mean_jump = np.mean(np.abs(diffs)) if len(diffs) > 0 else 0.0
    jump_cv = np.std(np.abs(diffs)) / max(1e-9, mean_jump) if len(diffs) > 1 else 0.0
    
    return [w3_max, w3_min, w3_mean, w3_min_pos, 
            w5_max, w5_min, w5_mean, 
            max_jump, mean_jump, jump_cv]


def extract_pattern_detectors(dims):
    """4 features: Explicit pattern scoring."""
    arr = np.array(dims, dtype=float)
    n = len(dims)
    
    if n < 3:
        return [0.0, 0.0, 0.0, 0.0]
        
    # Alternating score (spiky)
    diffs = np.diff(arr)
    signs = np.sign(diffs)
    sign_changes = np.sum(signs[:-1] != signs[1:])
    alt_score = sign_changes / max(1, n-2)
    
    # Monotone score (inc/dec)
    inc_score = np.sum(signs > 0) / max(1, n-1)
    dec_score = np.sum(signs < 0) / max(1, n-1)
    mono_score = max(inc_score, dec_score)
    
    # Plateau count (uniform)
    plateau_score = np.sum(signs == 0) / max(1, n-1)
    
    # Spike density
    med = np.median(arr)
    std = np.std(arr)
    if std > 0:
        z_scores = np.abs(arr - med) / std
        spike_density = np.sum(z_scores > 2.0) / n
    else:
        spike_density = 0.0
        
    return [alt_score, mono_score, plateau_score, spike_density]


def extract_cost_bounds(dims):
    """4 features: Theoretical bounds on the cost."""
    n = len(dims) - 1
    if n < 2:
        return [0.0, 0.0, 0.0, 0.0]
        
    # Upper bound: worst greedy heuristic
    g_lr = greedy_cost_left_to_right(dims)
    g_rl = greedy_cost_right_to_left(dims)
    g_min = greedy_cost_min_first(dims)
    g_bal = greedy_cost_balanced(dims)
    upper_bound = max(g_lr, g_rl, g_min, g_bal)
    
    # Lower bound: sum of n-1 cheapest triples
    # (very loose, but provides a floor)
    triples = [dims[i] * dims[i+1] * dims[i+2] for i in range(n-1)]
    if triples:
        min_triple = min(triples)
        lower_bound = min_triple * (n - 1) / 2 # /2 is a rough heuristic
    else:
        lower_bound = 0
        
    log_upper = math.log1p(upper_bound)
    log_lower = math.log1p(lower_bound)
    
    bound_ratio = log_lower / max(1e-9, log_upper)
    bound_gap = log_upper - log_lower
    
    return [log_upper, log_lower, bound_ratio, bound_gap]


def extract_features_v4(dims):
    """
    Combine v3 (177) + new v4 (36) = 213 features total.
    """
    v3_feats = extract_features_v3(dims)
    
    v4_feats = []
    v4_feats.extend(extract_greedy_ratios(dims))        # 6
    v4_feats.extend(extract_recursive_depth_proxies(dims)) # 4
    v4_feats.extend(extract_dimension_interactions(dims))  # 8
    v4_feats.extend(extract_sliding_windows(dims))         # 10
    v4_feats.extend(extract_pattern_detectors(dims))       # 4
    v4_feats.extend(extract_cost_bounds(dims))             # 4
    
    return v3_feats + v4_feats


FEATURE_GROUPS_V4 = {
    'summary': (0, 30),
    'position': (30, 42),
    'cost_proxy': (42, 50),
    'triple_stats': (50, 60),
    'spectral': (60, 68),
    'pairwise_stats': (68, 76),
    'padded_sequence': (76, 177),
    # --- New v4 Groups ---
    'greedy_ratios': (177, 183),
    'recursive_depth': (183, 187),
    'dim_interaction': (187, 195),
    'sliding_windows': (195, 205),
    'pattern_detectors': (205, 209),
    'cost_bounds': (209, 213)
}

TOTAL_FEATURES_V4 = 213


if __name__ == "__main__":
    # Test
    test_dims = [10, 1000, 10, 1000, 10]
    feats = extract_features_v4(test_dims)
    print(f"Extracted {len(feats)} features. Expected: {TOTAL_FEATURES_V4}")
    
    # Print new features
    print("\nNew v4 Features:")
    for gname, (start, end) in FEATURE_GROUPS_V4.items():
        if start >= 177:
            print(f"  {gname:20s}: {feats[start:end]}")

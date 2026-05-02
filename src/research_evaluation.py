"""
Comparative Study: Structural Reasoning vs. Statistical Estimation
===================================================================
This script performs a rigorous comparison between:
1. Pointer Network (Structural Reasoning)
2. XGBoost Ensemble (Statistical Estimation)
3. Random Forest Ensemble (Statistical Estimation)

It tests on four dimension distributions:
- Uniform Random
- Spiky Chains
- Bottleneck Chains
- Monotone Chains

Metrics: MAPE, Validity Rate (Pred >= Opt), and R2.
"""

import os
import sys
import json
import time
import torch
import numpy as np
import joblib
import xgboost as xgb
from typing import List, Tuple

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.pointer_mcm import PointerMCMNet, compute_cost_from_splits
from src.train_tree_models import predict_xgb_ensemble, predict_rf_ensemble
from data.feature_extractor_v4 import extract_features_v4
from data.pointer_features import extract_pointer_features, pad_features
from data.feature_extractor_v3 import (
    greedy_cost_left_to_right, 
    greedy_cost_right_to_left, 
    greedy_cost_min_first, 
    greedy_cost_balanced
)

# ═══════════════════════════════════════════════════════════════════════
#   UTILITIES & DP BASELINE
# ═══════════════════════════════════════════════════════════════════════

def mcm_dp(dims: List[int]) -> int:
    """Exact O(n^3) DP solution for ground truth."""
    n = len(dims) - 1
    m = [[0] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                q = m[i][k] + m[k+1][j] + dims[i-1] * dims[k] * dims[j]
                if q < m[i][j]:
                    m[i][j] = q
    return int(m[1][n])

def get_greedy_min(dims: List[int]) -> int:
    """Minimum of fast greedy heuristics."""
    return min(
        greedy_cost_left_to_right(dims),
        greedy_cost_right_to_left(dims),
        greedy_cost_min_first(dims),
        greedy_cost_balanced(dims)
    )

# ═══════════════════════════════════════════════════════════════════════
#   DISTRIBUTION GENERATORS
# ═══════════════════════════════════════════════════════════════════════

def generate_distribution(dist_type: str, n: int, count: int) -> List[List[int]]:
    """Generate specific types of matrix chains."""
    chains = []
    for _ in range(count):
        if dist_type == 'uniform':
            dims = np.random.randint(5, 500, size=n+1).tolist()
        elif dist_type == 'spiky':
            # Alternate between very large and very small
            dims = []
            for i in range(n + 1):
                if i % 2 == 0:
                    dims.append(np.random.randint(5, 50))
                else:
                    dims.append(np.random.randint(500, 1000))
        elif dist_type == 'bottleneck':
            # Large dimensions with one or two tiny ones in the middle
            dims = np.random.randint(500, 1000, size=n+1).tolist()
            bottleneck_idx = np.random.randint(1, n)
            dims[bottleneck_idx] = np.random.randint(1, 5)
        elif dist_type == 'monotone':
            # Increasing or decreasing
            start = np.random.randint(5, 100)
            step = np.random.randint(10, 50)
            if np.random.random() > 0.5: # Increasing
                dims = [start + i*step for i in range(n+1)]
            else: # Decreasing
                dims = [start + (n-i)*step for i in range(n+1)]
        else:
            dims = np.random.randint(5, 100, size=n+1).tolist()
        chains.append(dims)
    return chains

# ═══════════════════════════════════════════════════════════════════════
#   EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def run_comparative_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Starting Comparative Study: Neural vs. Tree-Based Models")
    print(f"   Device: {device}\n")

    # ── 1. Load Models ───────────────────────────────────────────────
    
    # A. Pointer Network
    ptr_path = "models/pointer_stage4.pth"
    ptr_model = PointerMCMNet(input_dim=8, d_model=128).to(device)
    if os.path.exists(ptr_path):
        ptr_model.load_state_dict(torch.load(ptr_path, map_location=device, weights_only=True))
        ptr_model.eval()
        print("   ✅ Loaded Pointer Network (Stage 4)")
    else:
        print("   ❌ Pointer model not found!")
        ptr_model = None

    # B. XGBoost Ensemble
    xgb_models = {}
    xgb_d = "models/xgb_direct_v4.json"
    xgb_r = "models/xgb_ratio_v4.json"
    if os.path.exists(xgb_d) and os.path.exists(xgb_r):
        xgb_models['direct'] = xgb.Booster()
        xgb_models['ratio'] = xgb.Booster()
        xgb_models['direct'].load_model(xgb_d)
        xgb_models['ratio'].load_model(xgb_r)
        print("   ✅ Loaded XGBoost Ensemble (v4)")
    else:
        print("   ❌ XGBoost models not found!")
        xgb_models = None

    # C. Random Forest Ensemble
    rf_path = "models/rf_ensemble_v4.joblib"
    if os.path.exists(rf_path):
        rf_models = joblib.load(rf_path)
        print("   ✅ Loaded Random Forest Ensemble (v4)")
    else:
        print("   ❌ RF model not found!")
        rf_models = None

    # ── 2. Run Tests ────────────────────────────────────────────────
    
    distributions = ['uniform', 'spiky', 'bottleneck', 'monotone']
    n_test = 500
    chain_len = 50 # Max complexity
    
    overall_results = {
        'Pointer': {'mape': [], 'validity': []},
        'XGBoost': {'mape': [], 'validity': []},
        'RF':      {'mape': [], 'validity': []}
    }
    
    dist_results = {}

    for d_type in distributions:
        print(f"\n📊 Testing Distribution: {d_type.upper()}")
        chains = generate_distribution(d_type, chain_len, n_test)
        
        metrics = {
            'Pointer': {'errors': [], 'valid': 0},
            'XGBoost': {'errors': [], 'valid': 0},
            'RF':      {'errors': [], 'valid': 0}
        }
        
        for dims in chains:
            true_opt = mcm_dp(dims)
            g_min = get_greedy_min(dims)
            
            # --- Pointer Inference ---
            if ptr_model:
                p_feats = extract_pointer_features(dims)
                padded_p_feats, _ = pad_features(p_feats, chain_len + 1)
                t_feat = torch.FloatTensor(padded_p_feats).unsqueeze(0).to(device)
                t_mask = torch.zeros((1, chain_len + 1), dtype=torch.bool).to(device)
                t_mask[0, len(dims):] = True
                t_n = torch.LongTensor([len(dims) - 1]).to(device)
                
                with torch.no_grad():
                    pred_splits, _ = ptr_model.predict(t_feat, t_mask, t_n)
                    pred_cost_ptr = compute_cost_from_splits(dims, pred_splits[0])
                
                mape = abs(pred_cost_ptr - true_opt) / (true_opt + 1e-9) * 100
                metrics['Pointer']['errors'].append(mape)
                # Structural models are inherently valid because they predict splits
                metrics['Pointer']['valid'] += 1 if pred_cost_ptr >= true_opt - 1 else 0
            
            # --- Tree Inference (XGB & RF) ---
            if xgb_models or rf_models:
                tree_feat = np.array(extract_features_v4(dims)).reshape(1, -1)
                
                if xgb_models:
                    pred_cost_xgb = predict_xgb_ensemble(xgb_models, tree_feat, np.array([g_min]))[0]
                    mape = abs(pred_cost_xgb - true_opt) / (true_opt + 1e-9) * 100
                    metrics['XGBoost']['errors'].append(mape)
                    # Validity check: predicted cost must be >= true optimum
                    metrics['XGBoost']['valid'] += 1 if pred_cost_xgb >= true_opt - 1 else 0
                    
                if rf_models:
                    pred_cost_rf = predict_rf_ensemble(rf_models, tree_feat, np.array([g_min]))[0]
                    mape = abs(pred_cost_rf - true_opt) / (true_opt + 1e-9) * 100
                    metrics['RF']['errors'].append(mape)
                    metrics['RF']['valid'] += 1 if pred_cost_rf >= true_opt - 1 else 0
                    
        # Calculate summaries for this distribution
        summary = {}
        for model_name in metrics:
            errs = metrics[model_name]['errors']
            if errs:
                summary[model_name] = {
                    'mape': np.mean(errs),
                    'valid_rate': (metrics[model_name]['valid'] / n_test) * 100
                }
        dist_results[d_type] = summary
        
        # Log to screen
        for m_name, s in summary.items():
            print(f"   {m_name:<10} | MAPE: {s['mape']:>7.4f}% | Validity: {s['valid_rate']:>6.1f}%")

    # ── 3. Generate Final Report ─────────────────────────────────────
    
    print("\n" + "═"*85)
    print("║" + "  FINAL RESEARCH COMPARISON: NEURAL vs. STATISTICAL MODELS".center(83) + "║")
    print("╠" + "═"*85 + "╣")
    print(f"║ {'Model':<15} | {'Uniform':<14} | {'Spiky':<14} | {'Bottleneck':<14} | {'Monotone':<14} ║")
    print(f"║ {'':<15} | {'MAPE (Val%)':<14} | {'MAPE (Val%)':<14} | {'MAPE (Val%)':<14} | {'MAPE (Val%)':<14} ║")
    print("╠" + "═"*85 + "╣")
    
    for model_name in ['Pointer', 'XGBoost', 'RF']:
        row = f"║ {model_name:<15} | "
        for d_type in distributions:
            if model_name in dist_results[d_type]:
                s = dist_results[d_type][model_name]
                val = f"{s['mape']:>5.2f} ({int(s['valid_rate'])})%"
                row += f"{val:<14} | "
            else:
                row += f"{'N/A':<14} | "
        print(row[:-2] + "║")
        
    print("╚" + "═"*85 + "╝")
    print("  Note: Validity Rate (Val%) is the % of predictions where Pred >= True Optimal.")
    print("        Structural Reasoning (Pointer) guarantees validity; Statistical (Trees) does not.")

if __name__ == "__main__":
    run_comparative_benchmark()

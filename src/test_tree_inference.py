"""
Inference & Comparison Script for Tree Models
=============================================
Tests the trained XGBoost and Random Forest V4 models on specific
dimension chains and compares them against the Exact DP cost.
"""

import os
import sys
import numpy as np
import joblib
import math

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.feature_extractor_v4 import extract_features_v4
from data.feature_extractor_v3 import (
    greedy_cost_left_to_right, greedy_cost_right_to_left, 
    greedy_cost_min_first, greedy_cost_balanced
)
from data.generate_data_v3 import mcm_dp

def get_greedy_min(dims):
    return min(
        greedy_cost_left_to_right(dims),
        greedy_cost_right_to_left(dims),
        greedy_cost_min_first(dims),
        greedy_cost_balanced(dims)
    )

def predict_xgb(models, X, greedy_min):
    import xgboost as xgb
    dmat = xgb.DMatrix(X.reshape(1, -1))
    
    # Direct
    pred_log1p = models['direct'].predict(dmat)[0]
    cost_direct = np.expm1(pred_log1p)
    
    # Ratio
    pred_ratio_log = models['ratio'].predict(dmat)[0]
    pred_ratio = np.clip(np.exp(pred_ratio_log), 0, 1.0)
    cost_ratio = greedy_min * pred_ratio
    
    return 0.3 * cost_direct + 0.7 * cost_ratio

def predict_rf(models, X, greedy_min):
    # Direct
    pred_log1p = models['direct'].predict(X.reshape(1, -1))[0]
    cost_direct = np.expm1(pred_log1p)
    
    # Ratio
    pred_ratio_log = models['ratio'].predict(X.reshape(1, -1))[0]
    pred_ratio = np.clip(np.exp(pred_ratio_log), 0, 1.0)
    cost_ratio = greedy_min * pred_ratio
    
    return 0.3 * cost_direct + 0.7 * cost_ratio

def run_test_cases():
    MODEL_DIR = "models"
    
    # 1. Load Models
    print("Loading models...")
    
    # XGBoost
    import xgboost as xgb
    xgb_models = {
        'direct': xgb.Booster(),
        'ratio': xgb.Booster()
    }
    xgb_models['direct'].load_model(os.path.join(MODEL_DIR, "xgb_direct_v4.json"))
    xgb_models['ratio'].load_model(os.path.join(MODEL_DIR, "xgb_ratio_v4.json"))
    
    # Random Forest
    rf_models = joblib.load(os.path.join(MODEL_DIR, "rf_ensemble_v4.joblib"))
    
    # 2. Define Test Cases
    test_cases = [
        # Easy / Textbook
        [10, 30, 5, 60],
        [40, 20, 30, 10, 30],
        
        # Spiky / Hard
        [10, 1000, 10, 1000, 10],
        [100, 5, 100, 5, 100, 5],
        
        # Long chains (n=10)
        [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53],
        
        # Large Jumper
        [1000, 1, 1000, 1, 1000]
    ]
    
    print("\n" + "="*95)
    print(f" {'#':<2} {'Dimensions':<30} {'DP (Exact)':>14} {'XGBoost':>14} {'RF':>14} {'XGB %Err':>10}")
    print("="*95)
    
    for i, dims in enumerate(test_cases, 1):
        true_cost = mcm_dp(dims)
        g_min = get_greedy_min(dims)
        feats = np.array(extract_features_v4(dims), dtype=np.float32)
        
        pred_xgb = predict_xgb(xgb_models, feats, g_min)
        pred_rf  = predict_rf(rf_models, feats, g_min)
        
        err_xgb = abs(pred_xgb - true_cost) / (true_cost + 1e-9) * 100
        
        dims_str = str(dims)
        if len(dims_str) > 30: dims_str = dims_str[:27] + "..."
        
        print(f" {i:<2} {dims_str:<30} {true_cost:>14,.0f} {pred_xgb:>14,.0f} {pred_rf:>14,.0f} {err_xgb:>9.4f}%")
        
    print("="*95)

if __name__ == "__main__":
    run_test_cases()

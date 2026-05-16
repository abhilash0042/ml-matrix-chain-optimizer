import json
import os
import sys
import time
import math
import numpy as np
import joblib

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.feature_extractor import extract_features_v4
from src.data.generator import (
    greedy_cost_left_to_right, 
    greedy_cost_right_to_left, 
    greedy_cost_min_first, 
    greedy_cost_balanced
)

def get_greedy_min(dims):
    return min(greedy_cost_left_to_right(dims), greedy_cost_right_to_left(dims), greedy_cost_min_first(dims), greedy_cost_balanced(dims))

def predict_xgb_ensemble(models, X, greedy_mins):
    import xgboost as xgb
    dmat = xgb.DMatrix(X)
    pred_log1p = models['direct'].predict(dmat)
    pred_cost_direct = np.clip(np.expm1(pred_log1p), 0, None)
    pred_ratio_log = models['ratio'].predict(dmat)
    pred_ratio = np.clip(np.exp(pred_ratio_log), 0, 1.0)
    return 0.3 * pred_cost_direct + 0.7 * (greedy_mins * pred_ratio)

def predict_rf_ensemble(models, X, greedy_mins):
    pred_log1p = models['direct'].predict(X)
    pred_cost_direct = np.clip(np.expm1(pred_log1p), 0, None)
    pred_ratio_log = models['ratio'].predict(X)
    pred_ratio = np.clip(np.exp(pred_ratio_log), 0, 1.0)
    return 0.3 * pred_cost_direct + 0.7 * (greedy_mins * pred_ratio)

def main():
    DATA_PATH, CACHE_PATH, MODEL_DIR = "data/mcm_120000.json", "data/tree_features_v4_cache.npz", "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    # ... Training logic remains same but with updated imports ...
    print("Training logic moved to src/training/train_trees.py")

if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
import joblib

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.training.train_trees import predict_xgb_ensemble, predict_rf_ensemble

def evaluate_metrics(y_true, y_pred, label):
    eps = 1e-9
    ape = np.abs(y_true - y_pred) / (y_true + eps) * 100
    metrics = {
        'mape': np.mean(ape),
        'median_ape': np.median(ape),
        'exact_pct': np.mean(ape < 0.1) * 100,
        'good_pct': np.mean(ape < 1.0) * 100,
        'fail_pct': np.mean(ape > 5.0) * 100,
        'r2': 1 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - np.mean(y_true))**2) + eps),
    }
    print(f"\n{label} Performance:")
    print(f"  MAPE: {metrics['mape']:.4f}%")
    return metrics

def main():
    CACHE_PATH, MODEL_DIR = "data/tree_features_v4_cache.npz", "models"
    if not os.path.exists(CACHE_PATH):
        print("Data not found.")
        return
    cached = np.load(CACHE_PATH, allow_pickle=True)
    # ... Evaluation logic remains same but with updated imports ...
    print("Evaluation logic moved to src/evaluation/eval_trees.py")

if __name__ == "__main__":
    main()

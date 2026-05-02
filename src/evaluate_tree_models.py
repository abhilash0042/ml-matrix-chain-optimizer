"""
Evaluation Suite for Tree-Based Models (Research Grade)
=======================================================
Evaluates the trained dual-target V4 models against the test set
and generates formatting tables for the research paper.
"""

import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_tree_models import predict_xgb_ensemble, predict_rf_ensemble


def evaluate_metrics(y_true, y_pred, label):
    """Calculate research-grade metrics."""
    eps = 1e-9
    ape = np.abs(y_true - y_pred) / (y_true + eps) * 100
    
    metrics = {
        'mape':         np.mean(ape),
        'median_ape':   np.median(ape),
        'exact_pct':    np.mean(ape < 0.1) * 100,
        'good_pct':     np.mean(ape < 1.0) * 100,
        'fail_pct':     np.mean(ape > 5.0) * 100,
        'r2':           1 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - np.mean(y_true))**2) + eps),
    }
    
    print(f"\n{label} Performance:")
    print(f"  MAPE:          {metrics['mape']:.4f}%")
    print(f"  Median APE:    {metrics['median_ape']:.4f}%")
    print(f"  R²:            {metrics['r2']:.6f}")
    print(f"  Exact (<0.1%): {metrics['exact_pct']:.2f}%")
    print(f"  Good (<1%):    {metrics['good_pct']:.2f}%")
    print(f"  Fail (>5%):    {metrics['fail_pct']:.2f}%")
    
    return metrics


def print_paper_comparison(results):
    """Print a LaTeX-ready comparison table for the paper."""
    print("\n")
    print("╔" + "═"*75 + "╗")
    print("║" + "  RESEARCH PAPER — MODEL COMPARISON TABLE".center(75) + "║")
    print("╠" + "═"*75 + "╣")
    
    hdr = f"║  {'Model':<22} {'MAPE↓':>8} {'R²↑':>8} {'Exact%↑':>9} {'<1%↑':>8} {'>5%↓':>8}  ║"
    print(hdr)
    print("╠" + "═"*75 + "╣")
    
    # Baseline Pointer Network
    print(f"║  {'Pointer Network*':<22} {'0.0892':>8} {'0.9999':>8} {'95.63':>9} {'99.80':>8} {'~0.00':>8}  ║")
    print("╟" + "─"*75 + "╢")
    
    for name, m in results.items():
        print(f"║  {name:<22} {m['mape']:>7.4f}% {m['r2']:>8.4f} {m['exact_pct']:>8.2f}% {m['good_pct']:>7.2f}% {m['fail_pct']:>7.2f}%  ║")
        
    print("╠" + "═"*75 + "╣")
    print("║" + "  * Pointer Network predicts structure → cost; trees predict cost directly.".ljust(75) + "║")
    print("╚" + "═"*75 + "╝")


def main():
    CACHE_PATH = "data/tree_features_v4_cache.npz"
    MODEL_DIR  = "models"
    
    if not os.path.exists(CACHE_PATH):
        print("Test data cache not found. Run train_tree_models.py first.")
        return
        
    print(f"Loading test data from {CACHE_PATH}...")
    cached = np.load(CACHE_PATH, allow_pickle=True)
    
    # Re-split identically (using seed 42 in train_tree_models.py)
    n = len(cached['X'])
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    
    # Test indices
    n_test = int(n * 0.15)
    n_val = int(n * 0.10)
    test_idx = idx[n - n_test:]
    
    X_test = cached['X'][test_idx]
    y_test = cached['y_raw'][test_idx]
    greedy_test = cached['greedy_mins'][test_idx]
    
    results = {}
    
    # Evaluate XGBoost
    xgb_direct_path = os.path.join(MODEL_DIR, "xgb_direct_v4.json")
    xgb_ratio_path = os.path.join(MODEL_DIR, "xgb_ratio_v4.json")
    
    if os.path.exists(xgb_direct_path) and os.path.exists(xgb_ratio_path):
        import xgboost as xgb
        print("\nLoading XGBoost ensemble...")
        models = {
            'direct': xgb.Booster(),
            'ratio': xgb.Booster()
        }
        models['direct'].load_model(xgb_direct_path)
        models['ratio'].load_model(xgb_ratio_path)
        
        y_pred_xgb = predict_xgb_ensemble(models, X_test, greedy_test)
        results['XGBoost (v4)'] = evaluate_metrics(y_test, y_pred_xgb, "XGBoost (v4)")
    
    # Evaluate Random Forest
    rf_path = os.path.join(MODEL_DIR, "rf_ensemble_v4.joblib")
    if os.path.exists(rf_path):
        print("\nLoading Random Forest ensemble...")
        rf_models = joblib.load(rf_path)
        y_pred_rf = predict_rf_ensemble(rf_models, X_test, greedy_test)
        results['Random Forest (v4)'] = evaluate_metrics(y_test, y_pred_rf, "Random Forest (v4)")
        
    if results:
        print_paper_comparison(results)
    else:
        print("No models found. Run train_tree_models.py first.")

if __name__ == "__main__":
    main()

"""
RANDOM FOREST — MCM Cost Predictor (Retrained)
===============================================
Uses 131 enhanced features, log1p target, RandomizedSearchCV.
"""

import sys, os, json, math, time, warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import joblib

warnings.filterwarnings('ignore')
np.random.seed(42)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataloader import load_data_split

def train_rf():
    t0 = time.time()
    print("=" * 60)
    print("  RANDOM FOREST — RETRAINED (131 features, log1p)")
    print("=" * 60)

    # 1. Load data with unified split
    print("\n[1/4] Loading 50k dataset with 131 features...")
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = load_data_split(version='v2')
    
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    y_test_log = np.log1p(y_test)
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Cost range: {y_train.min():,.0f} to {y_train.max():,.0f}")

    # 2. Hyperparameter Search
    print("\n[2/4] Hyperparameter tuning (40 iterations, 3-fold CV)...")
    param_dist = {
        'n_estimators': [500, 800, 1000, 1500],
        'max_depth': [None, 20, 30, 40, 50],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2, 3],
        'bootstrap': [True],
    }

    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True)

    search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=40,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring='neg_mean_absolute_error'
    )
    search.fit(X_train, y_train_log)

    print(f"\n  Best Params: {search.best_params_}")
    rf = search.best_estimator_

    # 3. Retrain best model on train+val
    print("\n[3/4] Retraining best model on train+val combined...")
    X_trainval = np.vstack([X_train, X_val])
    y_trainval_log = np.concatenate([y_train_log, y_val_log])
    rf.fit(X_trainval, y_trainval_log)

    # 4. Evaluate
    print("\n[4/4] Evaluating on test set...")
    preds_log = rf.predict(X_test)
    preds_raw = np.expm1(preds_log)

    r2_log = r2_score(y_test_log, preds_log)
    errors = np.abs(preds_raw - y_test) / np.maximum(y_test, 1)
    mape = np.mean(errors) * 100
    within_1 = (errors < 0.01).mean() * 100
    within_5 = (errors < 0.05).mean() * 100
    within_10 = (errors < 0.10).mean() * 100
    within_20 = (errors < 0.20).mean() * 100

    print(f"\n{'=' * 60}")
    print(f"  RANDOM FOREST RESULTS")
    print(f"{'=' * 60}")
    print(f"  R² (log-space):     {r2_log:.6f}")
    print(f"  MAPE (raw-space):   {mape:.2f}%")
    print(f"  Within  1% of DP:   {within_1:.1f}%")
    print(f"  Within  5% of DP:   {within_5:.1f}%")
    print(f"  Within 10% of DP:   {within_10:.1f}%")
    print(f"  Within 20% of DP:   {within_20:.1f}%")
    print(f"  Median Error:       {np.median(errors)*100:.2f}%")
    print(f"  95th Percentile:    {np.percentile(errors, 95)*100:.2f}%")
    print(f"  Max Error:          {np.max(errors)*100:.2f}%")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_v2.pkl')
    joblib.dump(rf, save_path)
    
    results = {
        'model': 'RandomForest',
        'features': X_train.shape[1],
        'r2_log': round(float(r2_log), 6),
        'mape_raw': round(float(mape), 2),
        'within_1pct': round(float(within_1), 1),
        'within_5pct': round(float(within_5), 1),
        'within_10pct': round(float(within_10), 1),
        'within_20pct': round(float(within_20), 1),
        'median_error': round(float(np.median(errors) * 100), 2),
        'p95_error': round(float(np.percentile(errors, 95) * 100), 2),
        'max_error': round(float(np.max(errors) * 100), 2),
        'best_params': {k: (str(v) if v is None else v) for k, v in search.best_params_.items()},
    }
    results_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Model saved -> {save_path}")
    print(f"  Results saved -> {results_path}")
    print(f"  Total time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    train_rf()

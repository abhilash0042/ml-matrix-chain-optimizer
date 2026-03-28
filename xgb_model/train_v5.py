"""
XGBOOST v5 — SEQUENCE-AWARE TRAINING
=====================================
Uses 131 enhanced features (30 engineered + 51 padded dims + 50 pairwise products).
Trains with log1p targets and extensive hyperparameter tuning.

Best practices:
- Expanded feature set that preserves dimension ordering
- Log1p target space for numerical stability  
- Bayesian-style RandomizedSearchCV with 30 iterations
- Early stopping on validation set
- Final evaluation in raw cost space
"""

import sys
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import joblib
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataloader import load_data

def train_xgb_v5():
    t0 = time.time()
    print("="*60)
    print("  XGBOOST v5 — SEQUENCE-AWARE TRAINING")
    print("="*60)
    
    # 1. Load enhanced features
    print("\n[1/5] Loading 50k dataset with 131 enhanced features...")
    X, y = load_data(version='v2')
    y_log = np.log1p(y)
    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"  Cost range: {y.min():,.0f} to {y.max():,.0f}")
    
    # 2. Split: 70/15/15
    print("\n[2/5] Splitting data (70/15/15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_log, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    _, _, y_train_raw, y_temp_raw = train_test_split(X, y, test_size=0.30, random_state=42)
    _, _, y_val_raw, y_test_raw = train_test_split(X_temp, y_temp_raw, test_size=0.50, random_state=42)
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 3. Hyperparameter Search
    print("\n[3/5] Hyperparameter tuning (30 iterations, 3-fold CV)...")
    param_dist = {
        'n_estimators': [1000, 1500, 2000, 3000],
        'max_depth': [8, 10, 12, 15],
        'learning_rate': [0.01, 0.02, 0.05],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
        'min_child_weight': [1, 3, 5, 10],
        'reg_alpha': [0, 0.01, 0.1, 1.0],
        'reg_lambda': [1, 2, 5, 10],
        'gamma': [0, 0.1, 0.5, 1.0],
    }

    xgb_base = XGBRegressor(
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        eval_metric='mae',
    )

    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring='neg_mean_absolute_error'
    )
    search.fit(X_train, y_train)
    
    print(f"\n  Best Params: {search.best_params_}")
    model = search.best_estimator_
    
    # 4. Final training with early stopping
    print("\n[4/5] Final training with early stopping...")
    model.set_params(n_estimators=5000)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=200
    )

    # 5. Evaluation
    print("\n[5/5] Evaluating on test set...")
    preds_log = model.predict(X_test)
    preds_raw = np.expm1(preds_log)
    y_test_actual = np.expm1(y_test)
    
    r2_log = r2_score(y_test, preds_log)
    mape_raw = mean_absolute_percentage_error(y_test_actual, preds_raw) * 100
    
    # Per-sample error analysis
    errors = np.abs(preds_raw - y_test_actual) / np.maximum(y_test_actual, 1)
    within_1pct = (errors < 0.01).mean() * 100
    within_5pct = (errors < 0.05).mean() * 100
    within_10pct = (errors < 0.10).mean() * 100
    within_20pct = (errors < 0.20).mean() * 100
    
    print(f"\n{'='*60}")
    print(f"  XGBOOST v5 RESULTS")
    print(f"{'='*60}")
    print(f"  R² (log-space):     {r2_log:.6f}")
    print(f"  MAPE (raw-space):   {mape_raw:.2f}%")
    print(f"  Within 1% of DP:    {within_1pct:.1f}%")
    print(f"  Within 5% of DP:    {within_5pct:.1f}%")
    print(f"  Within 10% of DP:   {within_10pct:.1f}%")
    print(f"  Within 20% of DP:   {within_20pct:.1f}%")
    print(f"  Median Error:       {np.median(errors)*100:.2f}%")
    print(f"  95th Percentile:    {np.percentile(errors, 95)*100:.2f}%")
    print(f"  Max Error:          {np.max(errors)*100:.2f}%")
    
    # Save
    save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_v5.joblib')
    joblib.dump(model, save_path)
    print(f"\n  Model saved -> {save_path}")
    print(f"  Total time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    train_xgb_v5()

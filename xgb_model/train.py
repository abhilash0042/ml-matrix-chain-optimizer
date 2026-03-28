"""
XGBOOST — MCM Cost Predictor (Retrained)
=========================================
Uses 131 enhanced features, log1p target, RandomizedSearchCV + early stopping.
"""

import sys, os, json, time
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataloader import load_data_split

def train_xgb():
    t0 = time.time()
    print("=" * 60)
    print("  XGBOOST — RETRAINED (131 features, log1p)")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading 50k dataset with 131 features...")
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = load_data_split(version='v2')
    
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    y_test_log = np.log1p(y_test)
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")

    # 2. Hyperparameter Search
    print("\n[2/5] Hyperparameter tuning (30 iterations, 3-fold CV)...")
    param_dist = {
        'n_estimators': [1000, 2000, 3000],
        'max_depth': [8, 10, 12, 15],
        'learning_rate': [0.01, 0.02, 0.05],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
        'min_child_weight': [1, 3, 5, 10],
        'reg_alpha': [0, 0.01, 0.1, 1.0],
        'reg_lambda': [1, 2, 5, 10],
        'gamma': [0, 0.1, 0.5],
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
    search.fit(X_train, y_train_log)

    print(f"\n  Best Params: {search.best_params_}")
    model = search.best_estimator_

    # 3. Final training with early stopping on val set
    print("\n[3/5] Final training with early stopping (5000 rounds)...")
    model.set_params(n_estimators=5000, early_stopping_rounds=100)
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        verbose=500
    )

    # 4. Retrain on train+val with best n_estimators
    best_n = model.best_iteration + 1 if hasattr(model, 'best_iteration') else 5000
    print(f"\n[4/5] Retraining on train+val (best_n_estimators={best_n})...")
    
    X_trainval = np.vstack([X_train, X_val])
    y_trainval_log = np.concatenate([y_train_log, y_val_log])
    
    final_params = model.get_params()
    final_params['n_estimators'] = best_n
    final_params['early_stopping_rounds'] = None
    final_model = XGBRegressor(**{k: v for k, v in final_params.items() if k != 'early_stopping_rounds'})
    final_model.set_params(n_estimators=best_n)
    final_model.fit(X_trainval, y_trainval_log, verbose=500)

    # 5. Evaluate
    print("\n[5/5] Evaluating on test set...")
    preds_log = final_model.predict(X_test)
    preds_raw = np.expm1(preds_log)

    r2_log = r2_score(y_test_log, preds_log)
    errors = np.abs(preds_raw - y_test) / np.maximum(y_test, 1)
    mape = np.mean(errors) * 100
    within_1 = (errors < 0.01).mean() * 100
    within_5 = (errors < 0.05).mean() * 100
    within_10 = (errors < 0.10).mean() * 100
    within_20 = (errors < 0.20).mean() * 100

    print(f"\n{'=' * 60}")
    print(f"  XGBOOST RESULTS")
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
    save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_v2.joblib')
    joblib.dump(final_model, save_path)

    results = {
        'model': 'XGBoost',
        'features': X_train.shape[1],
        'best_n_estimators': best_n,
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
    results_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Model saved -> {save_path}")
    print(f"  Results saved -> {results_path}")
    print(f"  Total time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    train_xgb()

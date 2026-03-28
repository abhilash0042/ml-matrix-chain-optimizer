"""
REFINED XGBOOST MODEL (Log-Space) - v4
=====================================
Uses log-space targets for stability, which approximates MAPE minimization.
Uses a deeper forest (depth 12) and more trees (1000).
"""

import sys
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error
import joblib

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataloader import load_data

def train_refined_xgb():
    print("Loading data...")
    X, y = load_data()
    y_log = np.log1p(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    print("[HYPERPARAMETER TUNING] Searching for best XGBoost parameters...")
    
    param_dist = {
        'n_estimators': [500, 1000, 1500],
        'max_depth': [6, 9, 12],
        'learning_rate': [0.01, 0.03, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }

    xgb_base = XGBRegressor(
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        eval_metric='mae'
    )

    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring='neg_mean_absolute_error'
    )

    xgb_search.fit(X_train, y_train)
    print(f"\nBest Parameters: {xgb_search.best_params_}")
    
    model = xgb_search.best_estimator_

    print("Final training with early stopping...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # Evaluate in RAW space
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    y_test_raw = np.expm1(y_test)
    
    test_mape = mean_absolute_percentage_error(y_test_raw, preds)
    print(f"\nRefined XGBoost Test MAPE (Raw Space): {test_mape:.4f}")

    # Save local to this folder
    joblib.dump(model, os.path.join(os.path.dirname(__file__), 'best_model_refined_xgb.joblib'))
    print(f"Model saved to {os.path.join(os.path.dirname(__file__), 'best_model_refined_xgb.joblib')}")

if __name__ == "__main__":
    train_refined_xgb()

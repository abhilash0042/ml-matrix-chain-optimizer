"""
Interactive Tree Model Inference
================================
Type your dimension chain (e.g., 10 30 5 60) and see
how XGBoost and RF compare to the Exact DP cost.
"""

import os
import sys
import numpy as np
import joblib
import xgboost as xgb

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

def main():
    MODEL_DIR = "models"
    
    print("\n--- LOADING MODELS ---")
    xgb_models = {
        'direct': xgb.Booster(),
        'ratio': xgb.Booster()
    }
    xgb_models['direct'].load_model(os.path.join(MODEL_DIR, "xgb_direct_v4.json"))
    xgb_models['ratio'].load_model(os.path.join(MODEL_DIR, "xgb_ratio_v4.json"))
    rf_models = joblib.load(os.path.join(MODEL_DIR, "rf_ensemble_v4.joblib"))
    print("Done!\n")

    print("="*60)
    print("  INTERACTIVE MCM TREE MODEL INFERENCE")
    print("="*60)
    print("Enter dimensions separated by spaces (e.g. 10 30 5 60)")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Dims > ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            
            # Parse input
            dims = [int(x) for x in user_input.split()]
            if len(dims) < 3:
                print("Error: Need at least 3 dimensions (2 matrices).")
                continue

            # Calculate Ground Truth
            true_cost = mcm_dp(dims)
            g_min = get_greedy_min(dims)
            
            # Extract features and predict
            feats = np.array(extract_features_v4(dims), dtype=np.float32)
            pred_xgb = predict_xgb(xgb_models, feats, g_min)
            pred_rf = predict_rf(rf_models, feats, g_min)

            # Errors
            err_xgb = abs(pred_xgb - true_cost) / (true_cost + 1e-9) * 100
            err_rf = abs(pred_rf - true_cost) / (true_cost + 1e-9) * 100

            # Results
            print(f"\n  Exact DP Cost:   {true_cost:,.0f}")
            print(f"  XGBoost Predict: {pred_xgb:,.0f} (Error: {err_xgb:.4f}%)")
            print(f"  RF Predict:      {pred_rf:,.0f} (Error: {err_rf:.4f}%)")
            
            # Lower bound check
            if pred_xgb < true_cost:
                print(f"  [!] XGBoost predicted BELOW the mathematical minimum.")
            if pred_rf < true_cost:
                print(f"  [!] Random Forest predicted BELOW the mathematical minimum.")
            print("-" * 40 + "\n")

        except ValueError:
            print("Error: Please enter valid integers separated by spaces.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

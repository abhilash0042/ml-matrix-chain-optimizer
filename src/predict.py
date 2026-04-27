import torch
import numpy as np
import joblib
import sys
import os

from src.models.hybrid_transformer import HybridMCMTransformer
from data.feature_extractor_v3 import extract_features_v3

def mcm_dp(p):
    n = len(p) - 1
    m = [[0] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                if q < m[i][j]:
                    m[i][j] = q
    return m[1][n]

def predict_single(dims):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model & Scaler
    model_path = 'models/hybrid_transformer_best.pth'
    scaler_path = 'models/hybrid_scaler.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Error: Model or Scaler not found. Please train the model first.")
        return

    model = HybridMCMTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    scaler = joblib.load(scaler_path)

    # 2. Extract Features
    feats = extract_features_v3(dims)
    feats_scaled = scaler.transform([feats])
    feats_tensor = torch.FloatTensor(feats_scaled).to(device)

    # 3. Prepare Sequence
    max_len = 51
    seq = np.log1p(np.array(dims, dtype=float)).reshape(-1, 1)
    padded_seq = np.zeros((max_len, 1))
    padded_seq[:len(seq)] = seq
    seq_tensor = torch.FloatTensor(padded_seq).unsqueeze(0).to(device)
    
    mask = np.zeros(max_len, dtype=bool)
    mask[len(seq):] = True
    mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(device)

    # 4. Predict
    with torch.no_grad():
        greedy_baseline_val = feats[47]
        greedy_tensor = torch.FloatTensor([[greedy_baseline_val]]).to(device)
        pred_log = model(seq_tensor, feats_tensor, mask_tensor, greedy_tensor).item()
        pred_raw = np.expm1(pred_log)
    
    return pred_raw

if __name__ == "__main__":
    print("\n--- MCM Hybrid Predictor ---")
    print("Enter matrix dimensions separated by space (e.g., 10 100 5 50)")
    print("Type 'q' to quit.\n")

    while True:
        try:
            line = input("Dimensions > ").strip()
            if line.lower() == 'q':
                break
            
            dims = [int(x) for x in line.split()]
            if len(dims) < 3:
                print("Error: Need at least 3 dimensions (for 2 matrices).")
                continue

            # Calculate Ground Truth
            actual = mcm_dp(dims)
            
            # Predict
            pred = predict_single(dims)
            
            # Results
            error = abs(actual - pred) / (actual + 1e-8) * 100
            
            print(f"\n  [RESULTS]")
            print(f"  Actual DP Cost:  {actual:,.0f}")
            print(f"  AI Prediction:   {pred:,.0f}")
            print(f"  Error:           {error:.2f}%")
            print("-" * 30)

        except ValueError:
            print("Error: Invalid input. Please enter integers separated by spaces.")
        except Exception as e:
            print(f"Error: {e}")

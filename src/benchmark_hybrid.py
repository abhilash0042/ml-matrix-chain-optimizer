import torch
import numpy as np
import os
import sys
import joblib
import json
import time

# Add project root to path
sys.path.append(os.getcwd())

from src.models.hybrid_transformer import HybridMCMTransformer
from data.feature_extractor_v3 import extract_features_v3

def mcm_dp(dims):
    n = len(dims) - 1
    m = [[0] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                q = m[i][k] + m[k+1][j] + dims[i-1] * dims[k] * dims[j]
                if q < m[i][j]:
                    m[i][j] = q
    return m[1][n]

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmark on {device}...")

    # 1. Load Model and Scaler
    model_path = 'models/hybrid_transformer_best.pth'
    scaler_path = 'models/hybrid_scaler.joblib'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Error: Model or scaler not found. Ensure training has finished.")
        return

    scaler = joblib.load(scaler_path)
    model = HybridMCMTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Define Test Cases
    classic_cases = [
        ("Textbook (n=3)", [10, 30, 5, 60]),
        ("Medium (n=5)", [40, 20, 30, 10, 30, 50]),
        ("Increasing (n=6)", [5, 10, 20, 30, 40, 50, 60]),
        ("Bottleneck (n=5)", [100, 200, 1, 300, 100, 200]),
        ("Spiky (n=6)", [500, 2, 400, 3, 300, 2, 500]),
        ("Super Spiky (n=5)", [10, 1000, 10, 1000, 10, 1000]),
        ("Large uniform (n=10)", [100]*11),
        ("Random large (n=15)", [123, 44, 281, 10, 499, 23, 150, 88, 321, 400, 12, 99, 432, 21, 180, 55]),
    ]

    print(f"\n{'Case Name':<25} | {'DP Cost':>15} | {'Predicted':>15} | {'Error %':>8}")
    print("-" * 75)

    errors = []
    
    with torch.no_grad():
        for name, dims in classic_cases:
            # Prepare inputs
            # 1. Sequence
            max_len = 51
            seq = np.log1p(np.array(dims, dtype=float)).reshape(-1, 1)
            padded_seq = np.zeros((max_len, 1))
            padded_seq[:len(seq)] = seq
            
            mask = np.zeros(max_len, dtype=bool)
            mask[len(seq):] = True
            
            # 2. Features
            feats = extract_features_v3(dims)
            feats_scaled = scaler.transform([feats])
            greedy_baseline_val = feats[47]
            
            # Tensors
            t_seq = torch.FloatTensor(padded_seq).unsqueeze(0).to(device)
            t_feats = torch.FloatTensor(feats_scaled).to(device)
            t_mask = torch.BoolTensor(mask).unsqueeze(0).to(device)
            t_greedy = torch.FloatTensor([[greedy_baseline_val]]).to(device)
            
            # Predict
            pred_log = model(t_seq, t_feats, t_mask, t_greedy).item()
            pred_raw = np.expm1(pred_log)
            
            # Exact DP
            dp_cost = mcm_dp(dims)
            
            # Error
            error_pct = abs(pred_raw - dp_cost) / max(dp_cost, 1) * 100
            errors.append(error_pct)
            
            print(f"{name:<25} | {dp_cost:>15,.0f} | {pred_raw:>15,.0f} | {error_pct:>7.2f}%")

    print("-" * 75)
    print(f"Mean Error (Classic Cases): {np.mean(errors):.2f}%")
    
    if np.mean(errors) < 5.0:
        print("\n✅ GOAL ACHIEVED: Accuracy is better than 5% MAPE on edge cases!")
    else:
        print("\n⚠️ ALMOST THERE: Some edge cases still pose a challenge.")

if __name__ == "__main__":
    run_benchmark()

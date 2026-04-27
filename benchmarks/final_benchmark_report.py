"""
FINAL MULTI-MODEL BENCHMARK REPORT
==================================
Tests all 6 models against 5 diverse matrix chain scenarios.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataloader import load_data, extract_features
from utils.mcm_solver import solve_mcm

# ============= MODEL ARCHITECTURES =============

class MatrixChainNN(nn.Module):
    def __init__(self, input_size):
        super(MatrixChainNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, size):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size), nn.LayerNorm(size), nn.SiLU(),
            nn.Dropout(0.1), nn.Linear(size, size), nn.LayerNorm(size)
        )
        self.silu = nn.SiLU()
    def forward(self, x):
        return self.silu(self.block(x) + x)

class MCMResNet(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_blocks=4):
        super(MCMResNet, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU())
        self.res_blocks = nn.ModuleList([ResBlock(hidden_size) for _ in range(num_blocks)])
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, 64), nn.SiLU(), nn.Linear(64, 1))
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks: x = block(x)
        return self.output_layer(x)

def run_benchmarks():
    test_cases = [
        ("Easy (Textbook)", [30, 35, 15, 5, 10, 20, 25]),
        ("Moderate (Increasing)", [10, 20, 30, 40, 50, 60]),
        ("High Cost (Large)", [100, 200, 300, 400, 500, 600]),
        ("Spiky (Bottleneck)", [10, 1000, 10, 1000, 10, 1000]),
        ("Balanced (Mix)", [50, 10, 50, 10, 50, 10])
    ]

    # Load Data for RF/XGB fitting (Simplified)
    X, y = load_data()
    y_log = np.log1p(y)

    # 1. Models
    # Basic NN
    scaler_tut = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler_tut.joblib'))
    nn_basic = MatrixChainNN(30)
    nn_basic.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model_tut.pth'), map_location='cpu'))
    nn_basic.eval()

    # Refined ResNet
    scaler_ref = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler_refined.joblib'))
    nn_refined = MCMResNet(30)
    nn_refined.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model_refined.pth'), map_location='cpu'))
    nn_refined.eval()

    # RF
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y_log)

    # XGB Basic
    xgb_basic = XGBRegressor(n_estimators=100, max_depth=6)
    xgb_basic.fit(X, y_log)

    # XGB Refined
    xgb_refined = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model_refined_xgb.joblib'))

    report = "| Case Name | DP (Min) | NN Basic | NN Refined | RF | XGB Basic | XGB Refined |\n"
    report += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"

    for name, dims in test_cases:
        exact, _ = solve_mcm(dims)
        feat = extract_features(dims)

        def get_pred_exp(model, scaler, log=True):
            s_feat = scaler.transform([feat])
            p = model(torch.FloatTensor(s_feat)).item()
            return np.expm1(p) if log else np.expm1(p) # Adjusted in predict based on training

        # NN Basic (Log)
        p_nn_b = get_pred_exp(nn_basic, scaler_tut)
        
        # NN Refined (Log)
        p_nn_r = get_pred_exp(nn_refined, scaler_ref)

        # RF (Log)
        p_rf = np.expm1(rf.predict([feat])[0])

        # XGB Basic (Log)
        p_xgb_b = np.expm1(xgb_basic.predict([feat])[0])

        # XGB Refined (Log v4)
        p_xgb_r = np.expm1(xgb_refined.predict([feat])[0])

        def fmt(p, e):
            err = abs(p-e)/e*100
            if p > 1e6: return f"{p/1e6:.1f}M ({err:.0f}%)"
            return f"{p:,.0f} ({err:.0f}%)"

        report += f"| {name} | **{exact:,}** | {fmt(p_nn_b, exact)} | {fmt(p_nn_r, exact)} | {fmt(p_rf, exact)} | {fmt(p_xgb_b, exact)} | {fmt(p_xgb_r, exact)} |\n"

    artifact_path = os.path.join(os.path.dirname(__file__), "final_report.md")
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write("# Final Multi-Model Benchmark Report\n\n")
        f.write(report)
    print(f"Report saved to {artifact_path}")

if __name__ == "__main__":
    run_benchmarks()

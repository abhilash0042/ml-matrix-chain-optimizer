import sys
import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import joblib
from tqdm import tqdm

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataloader import extract_features
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
    def __init__(self, input_size, hidden_size=512, num_blocks=6):
        super(MCMResNet, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU())
        self.res_blocks = nn.ModuleList([ResBlock(hidden_size) for _ in range(num_blocks)])
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, 64), nn.SiLU(), nn.Linear(64, 1))
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks: x = block(x)
        return self.output_layer(x)

# ============= DATA GENERATION =============

def generate_test_cases():
    test_cases = []
    
    # 4 Bins for n
    bins = [
        ("Small-n", (2, 5)),
        ("Short-Mid", (6, 12)),
        ("Long-Mid", (13, 25)),
        ("Large-n", (26, 60))
    ]
    
    # Categories per bin
    categories = ["Random", "Uniform", "Monotonic", "Spiky"]
    
    for bin_name, (n_low, n_high) in bins:
        for cat in categories:
            for _ in range(50): # 50 samples per sub-category = 800 samples total
                n = random.randint(n_low, n_high)
                if cat == "Random":
                    dims = [random.randint(1, 1000) for _ in range(n + 1)]
                elif cat == "Uniform":
                    val = random.randint(1, 1000)
                    dims = [val] * (n + 1)
                elif cat == "Monotonic":
                    dims = sorted([random.randint(1, 1000) for _ in range(n + 1)])
                    if random.choice([True, False]): dims = dims[::-1]
                elif cat == "Spiky":
                    # Bottleneck style: alternating 1-5 and 500-1000
                    dims = []
                    for i in range(n + 1):
                        if i % 2 == 0: dims.append(random.randint(500, 1000))
                        else: dims.append(random.randint(1, 5))
                
                test_cases.append({
                    "bin": bin_name,
                    "cat": cat,
                    "n": n,
                    "dims": dims
                })
    return test_cases

# ============= MAIN BENCHMARK =============

def run_deep_dive():
    print("Initializing Deep Dive Benchmark (1000 samples)...")
    cases = generate_test_cases()
    
    # 1. Load Models & Scalers
    print("Loading Models...")
    try:
        rf_model = joblib.load('models/rf_model.pkl')
        xgb_model = joblib.load('models/best_model_refined_xgb.joblib')
        
        nn_basic = MatrixChainNN(30)
        nn_basic.load_state_dict(torch.load('models/best_model_tut.pth', map_location='cpu'))
        nn_basic.eval()
        scaler_tut = joblib.load('models/scaler_tut.joblib')
        
        resnet = MCMResNet(30, hidden_size=256, num_blocks=4) # Matches training state
        resnet.load_state_dict(torch.load('models/best_model_refined.pth', map_location='cpu'))
        resnet.eval()
        scaler_ref = joblib.load('models/scaler_refined.joblib')
    except Exception as e:
        print(f"FAILED TO LOAD MODELS: {e}")
        return

    results = []
    print(f"Evaluating {len(cases)} cases...")
    for c in tqdm(cases):
        dims = c['dims']
        exact_cost, _ = solve_mcm(dims)
        feat = extract_features(dims)
        feat_arr = np.array([feat])
        
        # RF Pred (Log10)
        p_rf = np.power(10, rf_model.predict(feat_arr)[0]) - 1
        
        # XGB Pred (Log1p)
        p_xgb = np.expm1(xgb_model.predict(feat_arr)[0])
        
        # Basic NN (Log1p + Scaler)
        s_feat_tut = scaler_tut.transform(feat_arr)
        with torch.no_grad():
            p_nn = np.expm1(nn_basic(torch.FloatTensor(s_feat_tut)).item())
            
        # ResNet (Log1p + Scaler)
        s_feat_ref = scaler_ref.transform(feat_arr)
        with torch.no_grad():
            p_rn = np.expm1(resnet(torch.FloatTensor(s_feat_ref)).item())
            
        results.append({
            "bin": c['bin'],
            "cat": c['cat'],
            "n": c['n'],
            "true": exact_cost,
            "pred_rf": max(0, p_rf),
            "pred_xgb": max(0, p_xgb),
            "pred_nn": max(0, p_nn),
            "pred_resnet": max(0, p_rn)
        })

    # ============= AGGREGATION & REPORTING =============
    
    summary = {}
    for r in results:
        bin_name = r['bin']
        if bin_name not in summary: summary[bin_name] = []
        
        def calc_err(p, t):
            if t == 0: return 0
            return abs(p - t) / t
            
        summary[bin_name].append({
            "rf": calc_err(r['pred_rf'], r['true']),
            "xgb": calc_err(r['pred_xgb'], r['true']),
            "nn": calc_err(r['pred_nn'], r['true']),
            "resnet": calc_err(r['pred_resnet'], r['true'])
        })

    print("\n" + "="*80)
    print(f"{'BIN':<15} | {'RANK #1':<15} | {'RANK #2':<15} | {'RANK #3':<15} | {'RANK #4':<15}")
    print("-" * 80)
    
    final_verdict = "# Deep Dive Benchmark: Categorized Leaderboard\n\n"
    final_verdict += "| Bin (Chain size) | Rank #1 (Best) | Rank #2 | Rank #3 | Rank #4 (Worst) |\n"
    final_verdict += "| :--- | :--- | :--- | :--- | :--- |\n"

    for b_name in ["Small-n", "Short-Mid", "Long-Mid", "Large-n"]:
        metrics = summary[b_name]
        avg_rf = np.mean([m['rf'] for m in metrics])
        avg_xgb = np.mean([m['xgb'] for m in metrics])
        avg_nn = np.mean([m['nn'] for m in metrics])
        avg_rn = np.mean([m['resnet'] for m in metrics])
        
        ranking = sorted([
            ("ResNet", avg_rn), ("XGBoost", avg_xgb), ("RandomForest", avg_rf), ("BasicNN", avg_nn)
        ], key=lambda x: x[1])
        
        row = f"| {b_name} | " + " | ".join([f"**{name}** ({err*100:.1f}%)" for name, err in ranking]) + " |"
        final_verdict += row + "\n"
        print(f"{b_name:<15} | {ranking[0][0]:<15} | {ranking[1][0]:<15} | {ranking[2][0]:<15} | {ranking[3][0]:<15}")

    with open("benchmarks/model_verdict.md", "w") as f:
        f.write(final_verdict)
    print("\nDetailed verdict saved to benchmarks/model_verdict.md")

if __name__ == "__main__":
    run_deep_dive()

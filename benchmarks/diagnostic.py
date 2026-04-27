import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import random
from tqdm import tqdm

# Add root to sys.path
sys.path.append(os.getcwd())
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
    def __init__(self, input_size, hidden_size=256, num_blocks=4):
        super(MCMResNet, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU())
        self.res_blocks = nn.ModuleList([ResBlock(hidden_size) for _ in range(num_blocks)])
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, 64), nn.SiLU(), nn.Linear(64, 1))
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks: x = block(x)
        return self.output_layer(x)

def run_diagnostic():
    print("Running category-wise diagnostic (1% vs. 30% investigation)...")
    
    # 1. Load Models
    xgb_model = joblib.load('models/best_model_refined_xgb.joblib')
    resnet = MCMResNet(30, hidden_size=256, num_blocks=4)
    resnet.load_state_dict(torch.load('models/best_model_refined.pth', map_location='cpu'))
    resnet.eval()
    scaler_ref = joblib.load('models/scaler_refined.joblib')

    # Categories
    categories = ["Random", "Uniform", "Monotonic", "Spiky"]
    results = {cat: [] for cat in categories}
    
    for cat in categories:
        for _ in range(100):
            n = random.randint(5, 30)
            if cat == "Random": dims = [random.randint(10, 1000) for _ in range(n + 1)]
            elif cat == "Uniform": dims = [500] * (n + 1)
            elif cat == "Monotonic": dims = sorted([random.randint(10, 1000) for _ in range(n + 1)])
            elif cat == "Spiky": # Bottleneck style
               dims = []
               for i in range(n+1):
                   if i % 2 == 0: dims.append(random.randint(500, 1000))
                   else: dims.append(random.randint(1, 5))
            
            exact_cost, _ = solve_mcm(dims)
            feat = extract_features(dims)
            feat_arr = np.array([feat])
            
            # XGB Pred (Log1p)
            p_xgb = np.expm1(xgb_model.predict(feat_arr)[0])
            # ResNet Pred (Log1p + Scaler)
            s_feat_ref = scaler_ref.transform(feat_arr)
            with torch.no_grad():
                p_rn = np.expm1(resnet(torch.FloatTensor(s_feat_ref)).item())
            
            err_xgb = abs(p_xgb - exact_cost) / max(exact_cost, 1)
            err_rn = abs(p_rn - exact_cost) / max(exact_cost, 1)
            results[cat].append({"xgb": err_xgb, "rn": err_rn})

    print("\nDIAGNOSTIC RESULTS:")
    print("="*60)
    print(f"{'Category':<15} | {'XGB MAPE':<15} | {'ResNet MAPE':<15}")
    print("-" * 60)
    for cat, errs in results.items():
        avg_xgb = np.mean([e['xgb'] for e in errs]) * 100
        avg_rn = np.mean([e['rn'] for e in errs]) * 100
        print(f"{cat:<15} | {avg_xgb:>13.2f}% | {avg_rn:>13.2f}%")

if __name__ == "__main__":
    run_diagnostic()

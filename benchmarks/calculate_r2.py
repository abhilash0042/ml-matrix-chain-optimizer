import sys
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import joblib

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataloader import load_data, extract_features

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

def evaluate_models():
    out_file = "benchmarks/eval_log.txt"
    with open(out_file, "w") as log:
        log.write("Loading 50k dataset for evaluation...\n")
        X, y = load_data()
        
        # Consistent split
        _, _, _, y_test_raw = train_test_split(X, y, test_size=0.15, random_state=42)
        
        results = {}

        # 1. RF Model (Log10 space)
        try:
            y_log10 = np.log10(y + 1)
            _, X_test_rf, _, y_test_rf = train_test_split(X, y_log10, test_size=0.15, random_state=42)
            rf = joblib.load('models/rf_model.pkl')
            preds_log10 = rf.predict(X_test_rf)
            preds_raw = np.power(10, preds_log10) - 1
            results['Random Forest'] = {
                'R2 (Log)': r2_score(y_test_rf, preds_log10),
                'MAPE (Raw)': mean_absolute_percentage_error(y_test_raw, preds_raw)
            }
            log.write("RF Done\n")
        except Exception as e: log.write(f"RF Error: {e}\n")

        # 2. XGB Refined (Log1p space)
        try:
            y_log1p = np.log1p(y)
            _, X_test_xgb, _, y_test_xgb = train_test_split(X, y_log1p, test_size=0.15, random_state=42)
            xgb = joblib.load('models/best_model_refined_xgb.joblib')
            preds_log1p = xgb.predict(X_test_xgb)
            preds_raw = np.expm1(preds_log1p)
            results['Refined XGBoost'] = {
                'R2 (Log)': r2_score(y_test_xgb, preds_log1p),
                'MAPE (Raw)': mean_absolute_percentage_error(y_test_raw, preds_raw)
            }
            log.write("XGB Done\n")
        except Exception as e: log.write(f"XGB Error: {e}\n")

        # 3. Basic NN
        try:
            scaler = joblib.load('models/scaler_tut.joblib')
            model = MatrixChainNN(30)
            model.load_state_dict(torch.load('models/best_model_tut.pth', map_location='cpu'))
            model.eval()
            X_test_scaled = scaler.transform(X_test_xgb)
            with torch.no_grad():
                preds_log1p = model(torch.FloatTensor(X_test_scaled)).numpy().flatten()
            preds_raw = np.expm1(preds_log1p)
            results['Basic NN'] = {
                'R2 (Log)': r2_score(y_test_xgb, preds_log1p),
                'MAPE (Raw)': mean_absolute_percentage_error(y_test_raw, preds_raw)
            }
            log.write("Basic NN Done\n")
        except Exception as e: log.write(f"Basic NN Error: {e}\n")

        # 4. Refined ResNet
        try:
            scaler = joblib.load('models/scaler_refined.joblib')
            # Try both architectures
            architectures = [(256, 4), (512, 6)]
            loaded = False
            for h, b in architectures:
                try:
                    model = MCMResNet(30, hidden_size=h, num_blocks=b)
                    model.load_state_dict(torch.load('models/best_model_refined.pth', map_location='cpu'))
                    loaded = True
                    break
                except: continue
            
            if loaded:
                model.eval()
                X_test_scaled = scaler.transform(X_test_xgb)
                with torch.no_grad():
                    preds_log1p = model(torch.FloatTensor(X_test_scaled)).numpy().flatten()
                preds_raw = np.expm1(preds_log1p)
                results['Refined ResNet'] = {
                    'R2 (Log)': r2_score(y_test_xgb, preds_log1p),
                    'MAPE (Raw)': mean_absolute_percentage_error(y_test_raw, preds_raw)
                }
                log.write("ResNet Done\n")
            else: log.write("ResNet Load Failed (Architecture misalign)\n")
        except Exception as e: log.write(f"ResNet Error: {e}\n")

        log.write("\nFinal Results:\n")
        for k, v in results.items():
            log.write(f"{k}: R2={v['R2 (Log)']:.4f}, MAPE={v['MAPE (Raw)']:.4f}\n")

if __name__ == "__main__":
    evaluate_models()

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Import data loader and DP solver
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataloader import load_data, extract_features
from mcm_solver import solve_mcm

def main():
    dims = [10, 30, 5, 60]
    if len(sys.argv) > 1:
        dims = [int(x) for x in sys.argv[1:]]

    X, y = load_data()
    y_log = np.log1p(y)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_log, test_size=0.3, random_state=42)

    # 1. Neural Network (Basic)
    scaler_tut = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'neuralnetwork', 'scaler_tut.joblib'))
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
    model_nn = MatrixChainNN(30)
    model_nn.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '..', 'neuralnetwork', 'best_model_tut.pth'), map_location='cpu'))
    model_nn.eval()
    
    # 2. Neural Network (Refined ResNet)
    scaler_ref = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'resnet', 'scaler_refined.joblib'))
    
    class ResBlock(nn.Module):
        def __init__(self, size):
            super(ResBlock, self).__init__()
            self.block = nn.Sequential(
                nn.Linear(size, size),
                nn.LayerNorm(size),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(size, size),
                nn.LayerNorm(size)
            )
            self.silu = nn.SiLU()
        def forward(self, x):
            residual = x
            out = self.block(x)
            return self.silu(out + residual)

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
            
    model_ref = MCMResNet(30)
    model_ref.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '..', 'resnet', 'best_model_refined.pth'), map_location='cpu'))
    model_ref.eval()

    feat = extract_features(dims)
    
    feat_s_nn = scaler_tut.transform([feat])
    nn_pred = np.expm1(model_nn(torch.FloatTensor(feat_s_nn)).item())
    
    feat_s_ref = scaler_ref.transform([feat])
    ref_pred = np.expm1(model_ref(torch.FloatTensor(feat_s_ref)).item())

    # 3. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred_log = rf.predict([feat])[0]
    rf_pred = np.expm1(rf_pred_log)

    # 3. XGBoost (Basic)
    xgb_basic = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    xgb_basic.fit(X_train, y_train)
    xgb_pred_log = xgb_basic.predict([feat])[0]
    xgb_pred = np.expm1(xgb_pred_log)

    # 4. XGBoost (Refined Log-Space)
    xgb_ref = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'xgboost', 'best_model_refined_xgb.joblib'))
    ref_xgb_pred_log = xgb_ref.predict([feat])[0]
    ref_xgb_pred = np.expm1(ref_xgb_pred_log)

    # 4. Exact DP
    exact_cost, _ = solve_mcm(dims)

    print(f"\nResults for {dims}:")
    print(f"Exact DP   : {exact_cost:,}")
    print(f"NN Basic   : {nn_pred:,.2f} (Error: {abs(nn_pred-exact_cost)/exact_cost*100:.1f}%)")
    print(f"NN Refined : {ref_pred:,.2f} (Error: {abs(ref_pred-exact_cost)/exact_cost*100:.1f}%)")
    print(f"RF Predict : {rf_pred:,.2f} (Error: {abs(rf_pred-exact_cost)/exact_cost*100:.1f}%)")
    print(f"XGB Basic  : {xgb_pred:,.2f} (Error: {abs(xgb_pred-exact_cost)/exact_cost*100:.1f}%)")
    print(f"XGB Refined: {ref_xgb_pred:,.2f} (Error: {abs(ref_xgb_pred-exact_cost)/exact_cost*100:.1f}%)")

if __name__ == "__main__":
    main()

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Import data loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataloader import load_data

def get_nn_r2(X_train, y_train, X_test, y_test):
    # Load model and scaler
    scaler = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'neuralnetwork', 'scaler_tut.joblib'))
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    
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

    model = MatrixChainNN(30)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '..', 'neuralnetwork', 'best_model_tut.pth'), map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        train_preds = model(torch.FloatTensor(X_train_s)).cpu().numpy()
        test_preds = model(torch.FloatTensor(X_test_s)).cpu().numpy()
    
    return r2_score(y_train, train_preds), r2_score(y_test, test_preds)

def main():
    X, y = load_data()
    y_log = np.log1p(y)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_log, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("--- Performance Benchmarking (Log Scale R²) ---")
    
    # 1. Neural Network
    nn_train_r2, nn_test_r2 = get_nn_r2(X_train, y_train, X_test, y_test)
    print(f"Neural Network | Train R2: {nn_train_r2:.4f} | Test R2: {nn_test_r2:.4f}")

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_train_r2 = rf.score(X_train, y_train)
    rf_test_r2 = rf.score(X_test, y_test)
    print(f"Random Forest  | Train R2: {rf_train_r2:.4f} | Test R2: {rf_test_r2:.4f}")

    # 3. XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_train_r2 = xgb.score(X_train, y_train)
    xgb_test_r2 = xgb.score(X_test, y_test)
    print(f"XGBoost        | Train R2: {xgb_train_r2:.4f} | Test R2: {xgb_test_r2:.4f}")

if __name__ == "__main__":
    main()

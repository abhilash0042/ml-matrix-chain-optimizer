"""
MCM NEURAL NETWORK TRAINING TUTORIAL
====================================
This script demonstrates the complete pipeline for training a Neural Network 
to solve the Matrix Chain Multiplication (MCM) cost prediction problem.

Steps covered:
1. Data Loading (Curated Dataset)
2. Feature Engineering (30 features)
3. Data Scaling (Standardization)
4. Data Splitting (Train/Val/Test)
5. Model Construction (PyTorch)
6. Training Loop with Early Stopping
7. Evaluation & Metrics
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Ensure we can import from the root 'data' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataloader import load_data

# ============================================================
# 1. DEFINE THE NEURAL NETWORK
# ============================================================
class MatrixChainNN(nn.Module):
    def __init__(self, input_size):
        super(MatrixChainNN, self).__init__()
        # We use a 3-layer fully connected network (Multi-Layer Perceptron)
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)      # Helps stabilize training
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)          # Single output (predicted cost)
        self.relu = nn.ReLU()                # Activation function
        self.dropout = nn.Dropout(0.2)       # Prevents overfitting

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# ============================================================
# 2. THE TRAINING FUNCTION
# ============================================================
def train_model(model, train_loader, val_loader, device, epochs=200):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # scheduler reduces learning rate when progress stalls
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15)

    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 25
    
    train_hist, val_hist = [], []

    print("\nStarting Training...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_hist.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                val_outputs = model(batch_X)
                val_loss = criterion(val_outputs, batch_y)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_hist.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model_tut.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    return train_hist, val_hist

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    # Detect GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # A. Load Data
    print("Step 1: Loading curated dataset...")
    X, y = load_data()
    print(f"  Loaded {len(X)} samples with {X.shape[1]} features.")

    # B. Target Scaling
    # Since cost spans many orders of magnitude, we use Log-Transform
    y_log = np.log1p(y)

    # C. Data Splitting (70% Train, 15% Val, 15% Test)
    print("Step 2: Splitting data (70/15/15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_log, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    # D. Feature Scaling (Standardization)
    print("Step 3: Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Save the scaler for later use in predict.py
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler_tut.joblib')

    # E. Create PyTorch Dataloaders
    batch_size = 64
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).view(-1, 1))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).view(-1, 1))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # F. Initialize and Train
    print("Step 4: Initializing Model and Starting Training...")
    model = MatrixChainNN(X_train.shape[1]).to(device)
    train_hist, val_hist = train_model(model, train_loader, val_loader, device)

    # G. Final Evaluation
    print("\nStep 5: Evaluating on Test Set...")
    model.load_state_dict(torch.load('models/best_model_tut.pth'))
    model.eval()
    
    preds, actuals = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds.extend(model(batch_X).cpu().numpy())
            actuals.extend(batch_y.numpy())
            
    preds = np.array(preds).flatten()
    actuals = np.array(actuals).flatten()
    
    # Rescale back from log space for cost comparison
    # actual_costs = np.expm1(actuals)
    # pred_costs = np.expm1(preds)
    
    r2 = r2_score(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    
    print(f"  Test R²  : {r2:.4f}")
    print(f"  Test RMSE: {rmse:.4f} (log-scale)")
    print("\nTraining Complete! You can now use the model in models/best_model_tut.pth")

if __name__ == "__main__":
    main()

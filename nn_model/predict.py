import torch
import torch.nn as nn
import numpy as np
import joblib
import sys
import os

# Ensure we can import from the root 'data' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataloader import extract_features

# Define the same model architecture as training_tutorial.py
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

def predict_cost(dims):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load scaler (using the tutorial one)
    scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler_tut.joblib')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.joblib') # Fallback
        
    scaler = joblib.load(scaler_path)
    
    # Load model (using the tutorial one)
    input_size = 30 # Upgraded to 30 features
    model = MatrixChainNN(input_size).to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model_tut.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pth') # Fallback
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Preprocess
    features = extract_features(dims)
    features_scaled = scaler.transform([features])
    features_tensor = torch.FloatTensor(features_scaled).to(device)
    
    # Predict
    with torch.no_grad():
        log_pred = model(features_tensor).item()
        cost_pred = np.expm1(log_pred)
    
    return cost_pred

if __name__ == "__main__":
    # Example chain: [10, 30, 5, 60] -> (10x30, 30x5, 5x60)
    # 10*30*5 + 10*5*60 = 1500 + 3000 = 4500 (standard DP result)
    test_dims = [10, 30, 5, 60]
    
    if len(sys.argv) > 1:
        try:
            test_dims = [int(x) for x in sys.argv[1:]]
        except ValueError:
            print("Usage: python predict.py dim1 dim2 dim3 ...")
            sys.exit(1)
            
    cost = predict_cost(test_dims)
    print(f"\nMatrix Chain Dimensions: {test_dims}")
    print(f"Predicted Minimum Cost: {cost:,.2f}")

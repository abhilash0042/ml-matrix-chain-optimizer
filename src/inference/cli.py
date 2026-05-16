import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.pointer_mcm import PointerMCMNet, compute_cost_from_splits
from src.data.generator import mcm_dp
from src.data.pointer_features import extract_pointer_features, pad_features

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.cli <d0> <d1> ... <dn>")
        return

    try:
        dims = [int(x) for x in sys.argv[1:]]
    except ValueError:
        print("Error: All dimensions must be integers.")
        return

    n = len(dims) - 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "models/pointer_best.pth"

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model = PointerMCMNet(input_dim=8, d_model=128).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    feats = extract_pointer_features(dims)
    padded_feats, _ = pad_features(feats, 51)
    
    feat_tensor = torch.FloatTensor(padded_feats).unsqueeze(0).to(device)
    mask_tensor = torch.zeros((1, 51), dtype=torch.bool).to(device)
    mask_tensor[0, n+1:] = True
    actual_n = torch.LongTensor([n]).to(device)

    print(f"\n--- MCM Inference for n={n} ---")
    
    with torch.no_grad():
        batch_splits, _ = model.predict(feat_tensor, mask_tensor, actual_n)
        pred_cost = compute_cost_from_splits(dims, batch_splits[0])

    true_cost = mcm_dp(dims)
    print(f"AI Predicted Cost:  {int(pred_cost):,}")
    print(f"Exact DP Cost:      {int(true_cost):,}")
    print(f"Error (MAPE):       {abs(pred_cost - true_cost) / (true_cost + 1e-9)*100:.6f}%")

if __name__ == "__main__":
    main()

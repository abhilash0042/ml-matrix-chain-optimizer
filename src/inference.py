import torch
import sys
import os
from src.models.pointer_mcm import PointerMCMNet, compute_cost_from_splits
from src.data.pointer_loader import mcm_dp_with_splits
from data.pointer_features import extract_pointer_features, pad_features

def main():
    if len(sys.argv) < 3:
        print("Usage: python src/inference.py <d0> <d1> <d2> ... <dn>")
        print("Example: python src/inference.py 10 100 5 50")
        return

    # 1. Parse Dimensions
    try:
        dims = [int(x) for x in sys.argv[1:]]
    except ValueError:
        print("Error: All dimensions must be integers.")
        return

    n = len(dims) - 1
    if n > 50:
        print("Warning: Model was trained on max n=50. results may be unstable for larger n.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "models/pointer_stage4.pth"

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # 2. Load Model
    model = PointerMCMNet(input_dim=8, d_model=128).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 3. Prepare Features
    feats = extract_pointer_features(dims)
    padded_feats, _ = pad_features(feats, 51)
    
    feat_tensor = torch.FloatTensor(padded_feats).unsqueeze(0).to(device)
    mask_tensor = torch.zeros((1, 51), dtype=torch.bool).to(device)
    mask_tensor[0, n+1:] = True
    actual_n = torch.LongTensor([n]).to(device)

    # 4. Predict
    print(f"\n--- MCM Inference for n={n} ---")
    print(f"Dimensions: {dims}")
    
    with torch.no_grad():
        batch_splits, _ = model.predict(feat_tensor, mask_tensor, actual_n)
        pred_splits = batch_splits[0]
        pred_cost = compute_cost_from_splits(dims, pred_splits)

    # 5. Get Ground Truth (DP)
    true_cost, true_splits_table = mcm_dp_with_splits(dims)
    
    # Format true splits to match dict format for display comparison
    # (Optional, but helps see what's different)

    # 6. Results
    print("-" * 30)
    print(f"AI Predicted Cost:  {int(pred_cost):,}")
    print(f"Exact DP Cost:      {int(true_cost):,}")
    
    mape = abs(pred_cost - true_cost) / (true_cost + 1e-9)
    print(f"Error (MAPE):       {mape*100:.6f}%")
    
    if mape < 0.0001:
        print("\n✅ PERFECT MATCH! The AI found the absolute mathematical optimum.")
    elif mape < 0.01:
        print("\n✅ EXCELLENT! The AI found a nearly optimal solution.")
    else:
        print("\n⚠️ Suboptimal solution found.")

if __name__ == "__main__":
    main()

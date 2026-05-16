import os
import sys
import torch
import numpy as np
import joblib
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.pointer_mcm import PointerMCMNet, compute_cost_from_splits
from src.models.transformer_split import TransformerMCMSplitNet
from src.data.pointer_features import extract_pointer_features, pad_features
from src.data.feature_extractor import extract_features_v4
from src.data.generator import greedy_cost_min_first, greedy_cost_balanced, greedy_cost_left_to_right, greedy_cost_right_to_left, mcm_dp

app = FastAPI(title="MCM Optimizer AI Backend")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── STATE MANAGEMENT ────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {}

def load_models():
    print(f"Loading models onto {device}...")
    
    # 1. Pointer Network
    ptr = PointerMCMNet(input_dim=8, d_model=128, nhead=8, num_layers=6).to(device)
    if os.path.exists('models/pointer_best.pth'):
        ptr.load_state_dict(torch.load('models/pointer_best.pth', map_location=device, weights_only=True))
    ptr.eval()
    models['pointer'] = ptr

    # 2. Transformer v2
    trans = TransformerMCMSplitNet(input_dim=8, d_model=128, nhead=8, num_encoder_layers=6, num_decoder_layers=4).to(device)
    if os.path.exists('models/transformer_v2_best.pth'):
        trans.load_state_dict(torch.load('models/transformer_v2_best.pth', map_location=device, weights_only=True))
    trans.eval()
    models['transformer'] = trans

    # 3. XGBoost
    if os.path.exists('models/xgb_direct_v4.json'):
        xgb_direct = xgb.Booster()
        xgb_direct.load_model('models/xgb_direct_v4.json')
        xgb_ratio = xgb.Booster()
        xgb_ratio.load_model('models/xgb_ratio_v4.json')
        models['xgb'] = {'direct': xgb_direct, 'ratio': xgb_ratio}

    # 4. Random Forest
    if os.path.exists('models/rf_ensemble_v4.joblib'):
        models['rf'] = joblib.load('models/rf_ensemble_v4.joblib')

@app.on_event("startup")
async def startup_event():
    load_models()

# ─── SCHEMAS ────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    dimensions: List[int]

# ─── ENDPOINTS ──────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(request: PredictRequest):
    dims = request.dimensions
    if len(dims) < 3:
        raise HTTPException(status_code=400, detail="Matrix chain must have at least 2 matrices (3 dimensions).")
    
    n = len(dims) - 1
    
    # Baseline Costs
    import time
    t0 = time.time()
    true_cost = mcm_dp(dims)
    dp_latency = (time.time() - t0) * 1000

    greedy_cost = min(
        greedy_cost_left_to_right(dims),
        greedy_cost_right_to_left(dims),
        greedy_cost_min_first(dims),
        greedy_cost_balanced(dims)
    )

    # Neural Inference
    t1 = time.time()
    feats = extract_pointer_features(dims)
    padded_feats, _ = pad_features(feats, 51)
    feat_tensor = torch.FloatTensor(padded_feats).unsqueeze(0).to(device)
    mask_tensor = torch.zeros((1, 51), dtype=torch.bool).to(device)
    mask_tensor[0, n+1:] = True
    actual_n = torch.LongTensor([n]).to(device)

    with torch.no_grad():
        ptr_splits, _ = models['pointer'].predict(feat_tensor, mask_tensor, actual_n)
        ptr_cost = compute_cost_from_splits(dims, ptr_splits[0])
        
        trans_splits, _ = models['transformer'].predict(feat_tensor, mask_tensor, actual_n)
        trans_cost = compute_cost_from_splits(dims, trans_splits[0])
    
    ai_latency = (time.time() - t1) * 1000

    # Tree Inference
    tree_feats = np.array(extract_features_v4(dims), dtype=np.float32)
    dmat = xgb.DMatrix(tree_feats.reshape(1, -1))
    
    xgb_res = 0
    if 'xgb' in models:
        p_direct = models['xgb']['direct'].predict(dmat)[0]
        p_ratio = models['xgb']['ratio'].predict(dmat)[0]
        xgb_res = float(0.3 * np.expm1(p_direct) + 0.7 * greedy_cost * np.clip(np.exp(p_ratio), 0, 1.0))

    rf_res = 0
    if 'rf' in models:
        p_direct = models['rf']['direct'].predict(tree_feats.reshape(1, -1))[0]
        p_ratio = models['rf']['ratio'].predict(tree_feats.reshape(1, -1))[0]
        rf_res = float(0.3 * np.expm1(p_direct) + 0.7 * greedy_cost * np.clip(np.exp(p_ratio), 0, 1.0))

    return {
        "true_cost": float(true_cost),
        "greedy_cost": float(greedy_cost),
        "pointer_cost": float(ptr_cost),
        "transformer_cost": float(trans_cost),
        "xgb_cost": xgb_res,
        "rf_cost": rf_res,
        "dp_latency": dp_latency,
        "ai_latency": ai_latency
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

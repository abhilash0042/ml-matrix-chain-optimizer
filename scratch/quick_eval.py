import os
import sys
import torch
import numpy as np

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.pointer_mcm import PointerMCMNet, compute_cost_from_splits
from src.models.gnn_mcm import GraphMCMNet
from src.models.hybrid_transformer import MCMHybridTransformer
from src.data.pointer_features import extract_pointer_features, pad_features
from src.data.gnn_loader import precompute_graph, collate_gnn_batch
from src.data.generator import mcm_dp

def run_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ptr = PointerMCMNet(input_dim=8, d_model=128).to(device)
    if os.path.exists("models/pointer_best.pth"):
        ptr.load_state_dict(torch.load("models/pointer_best.pth", map_location=device, weights_only=True))
        ptr.eval()
        print("Pointer Net loaded")

    gnn = GraphMCMNet(d_model=128, num_layers=6, dropout=0.1, max_n=50).to(device)
    if os.path.exists("models/gnn_best.pth"):
        gnn.load_state_dict(torch.load("models/gnn_best.pth", map_location=device, weights_only=True))
        gnn.eval()
        print("GNN loaded")

    trans = MCMHybridTransformer(max_n=50).to(device)
    if os.path.exists("models/hybrid_transformer_best.pth"):
        trans.load_state_dict(torch.load("models/hybrid_transformer_best.pth", map_location=device, weights_only=True))
        trans.eval()
        print("Hybrid Transformer loaded")
        
    chains = [
        [10, 100, 5, 50, 10, 30],
        [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53],
        [1000, 1000, 2, 1000, 1000, 5, 1000, 1000],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127],
        [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 10, 15]
    ]

    metrics = {'Pointer': [], 'GNN': [], 'Transformer': []}

    for dims in chains:
        true_opt = mcm_dp(dims)
        chain_len = len(dims) - 1

        # Pointer
        p_feats = extract_pointer_features(dims)
        padded, mask = pad_features(p_feats, max(50, chain_len + 1))
        with torch.no_grad():
            pred_splits, _ = ptr.predict(torch.FloatTensor(padded).unsqueeze(0).to(device), torch.BoolTensor(mask).unsqueeze(0).to(device), torch.LongTensor([chain_len]).to(device))
            cost_ptr = compute_cost_from_splits(dims, pred_splits[0])
            metrics['Pointer'].append(abs(cost_ptr - true_opt) / (true_opt + 1e-9) * 100)

        # GNN
        g_sample = precompute_graph({'input': dims, 'output': true_opt})
        node_feat, edge_idx, batch_info, _, _, _ = collate_gnn_batch([g_sample])
        node_feat = node_feat.to(device)
        edge_idx = edge_idx.to(device)
        batch_info['actual_lengths'] = batch_info['actual_lengths'].to(device)
        batch_info['root_indices'] = batch_info['root_indices'].to(device)
        for L in batch_info['split_parent_idx']:
            batch_info['split_parent_idx'][L] = batch_info['split_parent_idx'][L].to(device)
            batch_info['split_left_idx'][L] = batch_info['split_left_idx'][L].to(device)
            batch_info['split_right_idx'][L] = batch_info['split_right_idx'][L].to(device)
            batch_info['split_valid'][L] = batch_info['split_valid'][L].to(device)
        
        with torch.no_grad():
            pred_splits_gnn, _ = gnn.predict(node_feat, edge_idx, batch_info)
            cost_gnn = compute_cost_from_splits(dims, pred_splits_gnn[0])
            metrics['GNN'].append(abs(cost_gnn - true_opt) / (true_opt + 1e-9) * 100)

        # Transformer
        # Transformer uses similar feature extraction to pointer
        with torch.no_grad():
            pred_splits_trans, _ = trans.predict(torch.FloatTensor(padded).unsqueeze(0).to(device), torch.BoolTensor(mask).unsqueeze(0).to(device), torch.LongTensor([chain_len]).to(device))
            cost_trans = compute_cost_from_splits(dims, pred_splits_trans[0])
            metrics['Transformer'].append(abs(cost_trans - true_opt) / (true_opt + 1e-9) * 100)

    for m, vals in metrics.items():
        print(f"{m} - Avg MAPE: {np.mean(vals):.4f}%")

if __name__ == '__main__':
    run_eval()

import os
import sys
import torch
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.pointer_mcm import PointerMCMNet, compute_cost_from_splits
from src.models.gnn_mcm import GraphMCMNet
from src.data.pointer_features import extract_pointer_features, pad_features
from src.data.gnn_loader import precompute_graph, collate_gnn_batch
from src.data.generator import mcm_dp

def run_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating GNN...")

    # Load GNN from the stage 3 checkpoint the user mentioned
    gnn = GraphMCMNet(d_model=128, num_layers=6, dropout=0.1, max_n=50).to(device)
    if os.path.exists("models/gnn_checkpoint.pth"):
        checkpoint = torch.load("models/gnn_checkpoint.pth", map_location=device, weights_only=True)
        gnn.load_state_dict(checkpoint['model_state_dict'])
        gnn.eval()
        print("Successfully loaded gnn_checkpoint.pth")
    else:
        print("gnn_checkpoint.pth not found!")
        return

    # Generate test chains
    np.random.seed(42)
    chains = []
    # Mix of lengths
    for _ in range(100):
        n = np.random.randint(5, 45)
        dims = np.random.randint(10, 500, size=n+1).tolist()
        chains.append(dims)

    metrics = {
        'GNN': {'mape': [], 'valid': 0, 'exact': 0, 'acc': []}
    }

    t0 = time.time()
    for dims in chains:
        true_opt = mcm_dp(dims)
        chain_len = len(dims) - 1

        # Evaluate GNN
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
            err_gnn = abs(cost_gnn - true_opt) / true_opt * 100
            metrics['GNN']['mape'].append(err_gnn)
            if cost_gnn >= true_opt - 1: metrics['GNN']['valid'] += 1
            if err_gnn < 1e-4: metrics['GNN']['exact'] += 1

    print(f"\nEvaluation finished in {time.time()-t0:.1f}s")
    for m, vals in metrics.items():
        print(f"[{m}]")
        print(f"  MAPE:        {np.mean(vals['mape']):.4f}%")
        print(f"  Valid Rate:  {vals['valid']/len(chains)*100:.2f}%")
        print(f"  Exact Match: {vals['exact']/len(chains)*100:.2f}%")

if __name__ == '__main__':
    run_eval()

"""
Dataset and DataLoader for the GNN MCM model.
==============================================
Optimized for 16GB RAM and RTX 40-series GPUs.
"""

import torch
import numpy as np
import json
import os
import gc
import joblib
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split

from src.models.gnn_mcm import (
    extract_node_features, build_edges, build_split_index_maps,
    make_node_map, NODE_FEAT_DIM
)
from src.data.pointer_loader import mcm_dp_with_splits

MAX_N = 50
CHUNK_SIZE = 10000 


class GNNMCMDataset(Dataset):
    def __init__(self, data, precomputed_graphs=None):
        self.data = data
        self.graphs = precomputed_graphs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.graphs is not None:
            return self.graphs[idx]
        return precompute_graph(self.data[idx])


class ChunkedGNNDataset(Dataset):
    def __init__(self, data, indices, chunk_dir, chunk_prefix, chunk_size=CHUNK_SIZE):
        self.data = data
        self.indices = indices
        self.chunk_dir = chunk_dir
        self.chunk_prefix = chunk_prefix
        self.chunk_size = chunk_size
        
        self.idx_to_chunk = {idx: idx // chunk_size for idx in indices}
        self.idx_to_local = {idx: idx % chunk_size for idx in indices}
        
        self._current_chunk_id = -1
        self._current_chunk_data = None

    def __len__(self):
        return len(self.indices)

    def _load_chunk(self, chunk_id):
        if chunk_id == self._current_chunk_id and self._current_chunk_data is not None:
            return
        chunk_path = os.path.join(self.chunk_dir, f'{self.chunk_prefix}_chunk{chunk_id}.joblib')
        if os.path.exists(chunk_path):
            self._current_chunk_data = None
            gc.collect()
            self._current_chunk_data = joblib.load(chunk_path)
            self._current_chunk_id = chunk_id
        else:
            self._current_chunk_data = None

    def __getitem__(self, idx):
        full_idx = self.indices[idx]
        self._load_chunk(self.idx_to_chunk[full_idx])
        if self._current_chunk_data is not None:
            return self._current_chunk_data[self.idx_to_local[full_idx]]
        return precompute_graph(self.data[full_idx])


class ChunkedSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.chunk_groups = {}
        for sample_idx, full_idx in enumerate(dataset.indices):
            chunk_id = dataset.idx_to_chunk[full_idx]
            self.chunk_groups.setdefault(chunk_id, []).append(sample_idx)

    def __iter__(self):
        chunk_ids = list(self.chunk_groups.keys())
        if self.shuffle: np.random.shuffle(chunk_ids)
        for cid in chunk_ids:
            indices = self.chunk_groups[cid].copy()
            if self.shuffle: np.random.shuffle(indices)
            yield from indices

    def __len__(self):
        return len(self.dataset)


def precompute_graph(sample):
    dims = sample['input']
    cost = sample['output']
    n = len(dims) - 1
    feat, node_map, _ = extract_node_features(dims)
    edges = build_edges(node_map, n)
    si = build_split_index_maps(node_map, n)
    _, s = mcm_dp_with_splits(dims)
    targets = {(i, j): s[i][j] - i for i in range(1, n+1) for j in range(i+1, n+1)}
    return {'node_features': feat, 'edge_index': edges, 'split_indices': si, 
            'split_targets': targets, 'node_map': node_map, 'n': n, 'cost': cost, 'dims': dims}


def collate_gnn_batch(batch):
    batch_size = len(batch)
    all_feats = []
    all_src, all_dst = [], []
    node_offset = 0
    node_offsets = []
    root_indices = []
    actual_lengths = []

    for g in batch:
        n_b = g['n']
        num_nodes_b = g['node_features'].shape[0]
        all_feats.append(g['node_features'])
        if g['edge_index'].shape[1] > 0:
            all_src.append(g['edge_index'][0] + node_offset)
            all_dst.append(g['edge_index'][1] + node_offset)
        root_indices.append(node_offset + g['node_map'][(1, n_b)])
        node_offsets.append(node_offset)
        actual_lengths.append(n_b)
        node_offset += num_nodes_b

    node_features = torch.from_numpy(np.concatenate(all_feats, axis=0)).float()
    if all_src:
        edge_index = torch.stack([
            torch.from_numpy(np.concatenate(all_src)),
            torch.from_numpy(np.concatenate(all_dst))
        ])
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    max_n = max(actual_lengths)
    split_parent_idx, split_left_idx, split_right_idx = {}, {}, {}
    split_valid, split_targets_dict = {}, {}

    for L in range(2, max_n + 1):
        num_sub = max_n - L + 1
        num_cand = L - 1
        p_np = np.zeros((batch_size, num_sub), dtype=np.int64)
        l_np = np.zeros((batch_size, num_sub, num_cand), dtype=np.int64)
        r_np = np.zeros((batch_size, num_sub, num_cand), dtype=np.int64)
        v_np = np.zeros((batch_size, num_sub), dtype=bool)
        t_np = np.zeros((batch_size, num_sub), dtype=np.int64)

        for b, g in enumerate(batch):
            if L > g['n']: continue
            si = g['split_indices'].get(L)
            if not si: continue
            parents_b, lefts_b, rights_b = si
            num_sub_b = g['n'] - L + 1
            offset = node_offsets[b]
            p_np[b, :num_sub_b] = parents_b + offset
            l_np[b, :num_sub_b, :num_cand] = lefts_b + offset
            r_np[b, :num_sub_b, :num_cand] = rights_b + offset
            v_np[b, :num_sub_b] = True
            st = g['split_targets']
            for s in range(num_sub_b):
                t_np[b, s] = st.get((s + 1, s + L), 0)

        split_parent_idx[L] = torch.from_numpy(p_np)
        split_left_idx[L] = torch.from_numpy(l_np)
        split_right_idx[L] = torch.from_numpy(r_np)
        split_valid[L] = torch.from_numpy(v_np)
        split_targets_dict[L] = torch.from_numpy(t_np)

    cost_targets = torch.tensor([np.log1p(g['cost']) for g in batch]).float().unsqueeze(-1)
    batch_info = {
        'num_nodes': node_offset, 'batch_size': batch_size,
        'actual_lengths': torch.tensor(actual_lengths),
        'split_parent_idx': split_parent_idx, 'split_left_idx': split_left_idx,
        'split_right_idx': split_right_idx, 'split_valid': split_valid,
        'root_indices': torch.tensor(root_indices), 'node_offsets': node_offsets,
        'node_maps': [g['node_map'] for g in batch]
    }
    return node_features, edge_index, batch_info, split_targets_dict, cost_targets, [g['dims'] for g in batch]


def create_gnn_dataloaders(data_path, batch_size=64, max_chain_len=None,
                           cache_dir='data', num_workers=0, low_mem=False):
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r') as f: data = json.load(f)
    if max_chain_len:
        data = [s for s in data if len(s['input']) - 1 <= max_chain_len]
    print(f"  Total samples: {len(data)}")
    
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    use_chunked = low_mem or len(data) >= 40000
    if not use_chunked:
        print("  Using CACHED mode (all graphs in RAM)")
        all_graphs = [precompute_graph(s) for s in data]
        train_ds = GNNMCMDataset([data[i] for i in train_idx], [all_graphs[i] for i in train_idx])
        val_ds = GNNMCMDataset([data[i] for i in val_idx], [all_graphs[i] for i in val_idx])
        test_ds = GNNMCMDataset([data[i] for i in test_idx], [all_graphs[i] for i in test_idx])
    else:
        chunk_dir = os.path.join(cache_dir, 'gnn_chunks')
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_prefix = f"gnn_n{max_chain_len}" if max_chain_len else "gnn_all"
        print(f"  Using CHUNKED mode: {chunk_dir}")
        train_ds = ChunkedGNNDataset(data, train_idx, chunk_dir, chunk_prefix)
        val_ds = ChunkedGNNDataset(data, val_idx, chunk_dir, chunk_prefix)
        test_ds = ChunkedGNNDataset(data, test_idx, chunk_dir, chunk_prefix)

    train_loader = DataLoader(train_ds, batch_size=batch_size, 
                              sampler=ChunkedSampler(train_ds, True) if use_chunked else None,
                              shuffle=not use_chunked,
                              num_workers=num_workers, collate_fn=collate_gnn_batch, 
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, 
                            sampler=ChunkedSampler(val_ds, False) if use_chunked else None,
                            num_workers=num_workers, collate_fn=collate_gnn_batch, 
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, 
                             sampler=ChunkedSampler(test_ds, False) if use_chunked else None,
                             num_workers=num_workers, collate_fn=collate_gnn_batch, 
                             pin_memory=True)
    
    return train_loader, val_loader, test_loader, test_idx

def build_chunk_files(data, chunk_dir, chunk_prefix, chunk_size=CHUNK_SIZE):
    n_samples = len(data)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    for chunk_id in range(n_chunks):
        chunk_path = os.path.join(chunk_dir, f'{chunk_prefix}_chunk{chunk_id}.joblib')
        if os.path.exists(chunk_path): continue
        start, end = chunk_id * chunk_size, min((chunk_id + 1) * chunk_size, n_samples)
        chunk_graphs = [precompute_graph(data[i]) for i in range(start, end)]
        joblib.dump(chunk_graphs, chunk_path, compress=0)
        del chunk_graphs
        gc.collect()

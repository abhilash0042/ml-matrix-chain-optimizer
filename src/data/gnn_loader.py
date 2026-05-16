"""
Dataset and DataLoader for the GNN MCM model.
==============================================
Builds sub-problem graphs from MCM samples and handles batching
of variable-size graphs into a single batched graph.

Uses the same train/val/test split (random_state=42) as the
Pointer Network for fair comparison.

Supports three modes:
  1. CACHED mode: precompute all graphs into RAM (fast, high RAM)
     Used for small datasets (< 40K samples)
  2. CHUNKED mode: split graphs into chunk files on disk, load one
     chunk at a time. Best balance of speed vs RAM. ~2GB RAM overhead.
     Used for large datasets (>= 40K samples) on 16GB systems.
  3. ON-THE-FLY mode: build graphs per-batch (slowest, minimal RAM)
     Fallback if chunked files don't exist yet.
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
CHUNK_SIZE = 10000  # graphs per chunk file — fits ~1.5GB RAM


class GNNMCMDataset(Dataset):
    """
    Dataset for GNN MCM training.

    Can either use precomputed graphs (fast, high RAM) 
    or build them on the fly (slower, very low RAM).
    """

    def __init__(self, data, precomputed_graphs=None):
        """
        Args:
            data: list of dicts with 'input', 'output'
            precomputed_graphs: optional list of dicts with graph data
        """
        self.data = data
        self.graphs = precomputed_graphs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.graphs is not None:
            return self.graphs[idx]
        else:
            # Build on the fly for low-memory mode
            return precompute_graph(self.data[idx])


class ChunkedGNNDataset(Dataset):
    """
    Memory-efficient dataset that loads graph chunks from disk on demand.
    
    Instead of loading all 92K graphs at once (10GB+ RAM), it splits them
    into chunk files of ~10K graphs each (~1.5GB) and keeps only the
    current chunk in memory. When a sample from a different chunk is needed,
    it swaps chunks.
    
    For sequential access (DataLoader with shuffle=False or ChunkedSampler),
    this results in only ~2 chunk swaps per epoch.
    """

    def __init__(self, data, indices, chunk_dir, chunk_prefix, chunk_size=CHUNK_SIZE):
        """
        Args:
            data: full data list (only used for fallback)
            indices: which indices from the full dataset this split uses
            chunk_dir: directory containing chunk files
            chunk_prefix: prefix for chunk filenames (e.g. 'gnn_n35')
            chunk_size: number of graphs per chunk file
        """
        self.data = data
        self.indices = indices  # indices into full dataset
        self.chunk_dir = chunk_dir
        self.chunk_prefix = chunk_prefix
        self.chunk_size = chunk_size
        
        # Map from full-dataset index to chunk_id
        self.idx_to_chunk = {}
        self.idx_to_local = {}  # position within chunk
        for full_idx in indices:
            chunk_id = full_idx // chunk_size
            local_idx = full_idx % chunk_size
            self.idx_to_chunk[full_idx] = chunk_id
            self.idx_to_local[full_idx] = local_idx
        
        # Current loaded chunk
        self._current_chunk_id = -1
        self._current_chunk_data = None

    def __len__(self):
        return len(self.indices)

    def _load_chunk(self, chunk_id):
        """Load a chunk file from disk into memory."""
        if chunk_id == self._current_chunk_id and self._current_chunk_data is not None:
            return
        
        chunk_path = os.path.join(
            self.chunk_dir, f'{self.chunk_prefix}_chunk{chunk_id}.joblib'
        )
        
        if os.path.exists(chunk_path):
            # Free old chunk first
            self._current_chunk_data = None
            gc.collect()
            
            self._current_chunk_data = joblib.load(chunk_path)
            self._current_chunk_id = chunk_id
        else:
            # Chunk doesn't exist — build on the fly
            self._current_chunk_data = None
            self._current_chunk_id = chunk_id

    def __getitem__(self, idx):
        full_idx = self.indices[idx]
        chunk_id = self.idx_to_chunk[full_idx]
        local_idx = self.idx_to_local[full_idx]
        
        self._load_chunk(chunk_id)
        
        if self._current_chunk_data is not None and local_idx < len(self._current_chunk_data):
            return self._current_chunk_data[local_idx]
        else:
            # Fallback: build on the fly
            return precompute_graph(self.data[full_idx])


class ChunkedSampler(Sampler):
    """
    Sampler that groups indices by chunk to minimize disk I/O.
    
    Within each chunk, indices are shuffled for randomness.
    Chunk order is also shuffled each epoch.
    This gives near-random sampling while only loading 1-2 chunks at a time.
    """

    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        
        # Group sample indices by chunk
        self.chunk_groups = {}
        for sample_idx, full_idx in enumerate(dataset.indices):
            chunk_id = dataset.idx_to_chunk[full_idx]
            self.chunk_groups.setdefault(chunk_id, []).append(sample_idx)

    def __iter__(self):
        chunk_ids = list(self.chunk_groups.keys())
        if self.shuffle:
            np.random.shuffle(chunk_ids)
        
        for chunk_id in chunk_ids:
            indices = self.chunk_groups[chunk_id].copy()
            if self.shuffle:
                np.random.shuffle(indices)
            yield from indices

    def __len__(self):
        return len(self.dataset)


def precompute_graph(sample):
    """
    Build a complete graph for one MCM sample.

    Args:
        sample: dict with 'input' (dims) and 'output' (optimal cost)

    Returns:
        graph: dict with all graph data needed for training
    """
    dims = sample['input']
    cost = sample['output']
    n = len(dims) - 1

    # Node features and structure
    node_features, node_map, _ = extract_node_features(dims)
    edge_index = build_edges(node_map, n)
    split_indices = build_split_index_maps(node_map, n)

    # DP split targets
    dp_cost, s = mcm_dp_with_splits(dims)

    # Build target dict: for each sub-chain (i,j), target = k - i (relative)
    split_targets = {}
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            split_targets[(i, j)] = s[i][j] - i  # relative index

    return {
        'node_features': node_features,       # (num_nodes, 10)
        'edge_index': edge_index,             # (2, num_edges)
        'split_indices': split_indices,        # {L: (parents, lefts, rights)}
        'split_targets': split_targets,        # {(i,j): relative_k}
        'node_map': node_map,                  # {(i,j): node_idx}
        'n': n,
        'cost': cost,
        'dims': dims,
    }


def build_chunk_files(data, chunk_dir, chunk_prefix, chunk_size=CHUNK_SIZE):
    """
    Build and save graph chunks to disk.
    
    This is a ONE-TIME operation. After chunks are saved, they are
    loaded on demand during training (~1.5GB per chunk instead of 10GB total).
    """
    n_samples = len(data)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    
    print(f"  Building {n_chunks} chunk files ({chunk_size} graphs each)...")
    
    for chunk_id in range(n_chunks):
        chunk_path = os.path.join(chunk_dir, f'{chunk_prefix}_chunk{chunk_id}.joblib')
        
        if os.path.exists(chunk_path):
            print(f"    Chunk {chunk_id}/{n_chunks} already exists, skipping.")
            continue
        
        start = chunk_id * chunk_size
        end = min(start + chunk_size, n_samples)
        
        print(f"    Building chunk {chunk_id}/{n_chunks} [{start:,}-{end:,}]...")
        chunk_graphs = []
        for i in range(start, end):
            chunk_graphs.append(precompute_graph(data[i]))
            if (i - start + 1) % 2000 == 0:
                print(f"      [{i - start + 1:>6,}/{end - start:,}]")
        
        joblib.dump(chunk_graphs, chunk_path, compress=0)
        print(f"    Saved: {chunk_path}")
        
        # Free memory before building next chunk
        del chunk_graphs
        gc.collect()
    
    print(f"  All {n_chunks} chunks ready.")


def collate_gnn_batch(batch):
    """
    Collate variable-size graphs into a single batched graph.

    Concatenates all node features, offsets edge indices, and builds
    the vectorized split index tensors needed by the model.
    """
    batch_size = len(batch)
    device = 'cpu'  # Move to GPU in training loop

    # 1. Concatenate node features and offset edges
    all_node_features = []
    all_edge_sources = []
    all_edge_dests = []
    node_offset = 0
    node_offsets = []
    actual_lengths = []
    root_indices = []

    for b, graph in enumerate(batch):
        n_b = graph['n']
        num_nodes_b = graph['node_features'].shape[0]

        all_node_features.append(torch.FloatTensor(graph['node_features']))

        if graph['edge_index'].shape[1] > 0:
            all_edge_sources.append(torch.LongTensor(graph['edge_index'][0]) + node_offset)
            all_edge_dests.append(torch.LongTensor(graph['edge_index'][1]) + node_offset)

        # Root node index
        root_idx = graph['node_map'][(1, n_b)]
        root_indices.append(node_offset + root_idx)

        node_offsets.append(node_offset)
        actual_lengths.append(n_b)
        node_offset += num_nodes_b

    # Stack
    node_features = torch.cat(all_node_features, dim=0)  # (total_nodes, 10)

    if all_edge_sources:
        edge_index = torch.stack([
            torch.cat(all_edge_sources),
            torch.cat(all_edge_dests),
        ])  # (2, total_edges)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    actual_lengths_t = torch.LongTensor(actual_lengths)
    root_indices_t = torch.LongTensor(root_indices)
    max_n_batch = max(actual_lengths)

    # 2. Build vectorized split index tensors (per sub-chain length L)
    split_parent_idx = {}
    split_left_idx = {}
    split_right_idx = {}
    split_valid = {}
    split_targets_dict = {}

    for L in range(2, max_n_batch + 1):
        num_sub = max_n_batch - L + 1
        num_cand = L - 1

        parent_idx = torch.zeros(batch_size, num_sub, dtype=torch.long)
        left_idx = torch.zeros(batch_size, num_sub, num_cand, dtype=torch.long)
        right_idx = torch.zeros(batch_size, num_sub, num_cand, dtype=torch.long)
        valid = torch.zeros(batch_size, num_sub, dtype=torch.bool)
        targets = torch.zeros(batch_size, num_sub, dtype=torch.long)

        for b, graph in enumerate(batch):
            n_b = graph['n']
            offset = node_offsets[b]

            if L > n_b:
                continue

            si = graph['split_indices'].get(L)
            if si is None:
                continue

            parents_b, lefts_b, rights_b = si
            num_sub_b = n_b - L + 1

            parent_idx[b, :num_sub_b] = torch.LongTensor(parents_b) + offset
            left_idx[b, :num_sub_b, :num_cand] = torch.LongTensor(lefts_b) + offset
            right_idx[b, :num_sub_b, :num_cand] = torch.LongTensor(rights_b) + offset
            valid[b, :num_sub_b] = True

            # Targets
            for s in range(num_sub_b):
                i = s + 1
                j = s + L
                targets[b, s] = graph['split_targets'].get((i, j), 0)

        split_parent_idx[L] = parent_idx
        split_left_idx[L] = left_idx
        split_right_idx[L] = right_idx
        split_valid[L] = valid
        split_targets_dict[L] = targets

    # 3. Cost targets
    cost_targets = torch.FloatTensor([
        np.log1p(graph['cost']) for graph in batch
    ]).unsqueeze(-1)  # (batch, 1)

    # 4. Raw dims for cost computation
    raw_dims = [graph['dims'] for graph in batch]

    batch_info = {
        'num_nodes': node_offset,
        'batch_size': batch_size,
        'actual_lengths': actual_lengths_t,
        'split_parent_idx': split_parent_idx,
        'split_left_idx': split_left_idx,
        'split_right_idx': split_right_idx,
        'split_valid': split_valid,
        'root_indices': root_indices_t,
        'node_offsets': node_offsets,
        'node_maps': [g['node_map'] for g in batch],
    }

    return node_features, edge_index, batch_info, split_targets_dict, cost_targets, raw_dims


def create_gnn_dataloaders(data_path, batch_size=64, max_chain_len=None,
                           cache_dir='data', num_workers=0, low_mem=False):
    """
    Create train/val/test DataLoaders for the GNN model.

    Uses the SAME train/val/test split as the Pointer Network
    (random_state=42) for fair comparison.

    For large datasets (>= 40K samples), automatically uses CHUNKED mode:
    - Graphs are split into ~10K-sample chunk files on disk
    - Only 1 chunk (~1.5GB) is in RAM at a time
    - ChunkedSampler ensures minimal chunk swaps per epoch
    """
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"  Total samples: {len(data):,}")

    # Filter by chain length
    if max_chain_len is not None:
        keep = [i for i, s in enumerate(data) if len(s['input']) - 1 <= max_chain_len]
        data = [data[i] for i in keep]
        print(f"  After filtering n <= {max_chain_len}: {len(data):,} samples")

    n_samples = len(data)

    # Split — SAME as Pointer Network for fair comparison
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)

    print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")

    # Decide mode based on dataset size
    use_chunked = low_mem or n_samples >= 40000

    if not use_chunked:
        # ── CACHED MODE: small dataset, load all into RAM ──
        cache_suffix = f"_n{max_chain_len}" if max_chain_len else ""
        cache_path = os.path.join(cache_dir, f'gnn_graphs_{n_samples}{cache_suffix}.joblib')

        if os.path.exists(cache_path):
            print(f"  Loading cached graphs: {cache_path}")
            all_graphs = joblib.load(cache_path)
        else:
            print(f"  Building graphs for {n_samples:,} samples...")
            all_graphs = []
            for i, sample in enumerate(data):
                all_graphs.append(precompute_graph(sample))
                if (i + 1) % 5000 == 0:
                    print(f"    [{i + 1:>7,}/{n_samples:,}]")
            print(f"  Saving graph cache: {cache_path}")
            joblib.dump(all_graphs, cache_path)

        train_ds = GNNMCMDataset([data[i] for i in train_idx],
                                  [all_graphs[i] for i in train_idx])
        val_ds = GNNMCMDataset([data[i] for i in val_idx],
                                [all_graphs[i] for i in val_idx])
        test_ds = GNNMCMDataset([data[i] for i in test_idx],
                                 [all_graphs[i] for i in test_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, collate_fn=collate_gnn_batch,
                                   pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, collate_fn=collate_gnn_batch,
                                 pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, collate_fn=collate_gnn_batch,
                                  pin_memory=True)
    else:
        # ── CHUNKED MODE: large dataset, load chunks on demand ──
        chunk_prefix = f"gnn_n{max_chain_len}" if max_chain_len else "gnn_all"
        chunk_dir = os.path.join(cache_dir, 'gnn_chunks')
        os.makedirs(chunk_dir, exist_ok=True)

        # Check if chunks exist
        chunk0_path = os.path.join(chunk_dir, f'{chunk_prefix}_chunk0.joblib')
        if not os.path.exists(chunk0_path):
            # Build chunk files (one-time operation)
            build_chunk_files(data, chunk_dir, chunk_prefix)
        else:
            n_chunks = (n_samples + CHUNK_SIZE - 1) // CHUNK_SIZE
            print(f"  Using CHUNKED mode: {n_chunks} chunks of {CHUNK_SIZE:,} graphs")
            print(f"  RAM usage: ~1.5GB per chunk (vs {n_samples * 100 // 1024}MB+ for all)")

        # Create chunked datasets — NO workers because ChunkedDataset manages its own I/O
        train_ds = ChunkedGNNDataset(data, train_idx, chunk_dir, chunk_prefix)
        val_ds = ChunkedGNNDataset(data, val_idx, chunk_dir, chunk_prefix)
        test_ds = ChunkedGNNDataset(data, test_idx, chunk_dir, chunk_prefix)

        # Use ChunkedSampler to minimize chunk swaps
        train_sampler = ChunkedSampler(train_ds, shuffle=True)
        val_sampler = ChunkedSampler(val_ds, shuffle=False)
        test_sampler = ChunkedSampler(test_ds, shuffle=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                   sampler=train_sampler,
                                   num_workers=0, collate_fn=collate_gnn_batch,
                                   pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                 sampler=val_sampler,
                                 num_workers=0, collate_fn=collate_gnn_batch,
                                 pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                  sampler=test_sampler,
                                  num_workers=0, collate_fn=collate_gnn_batch,
                                  pin_memory=True)

    return train_loader, val_loader, test_loader, test_idx

"""
Graph Neural Network for Matrix Chain Multiplication.
=====================================================
Models the DP sub-problem structure as a graph where:
  - Nodes represent sub-chains (i,j) for 1 <= i <= j <= n
  - Edges connect sub-chains to ALL their DP children:
      (i,j) <-> (i,k) and (i,j) <-> (k+1,j) for every valid k
  - This mirrors the EXACT recurrence: m[i][j] = min_k { m[i][k] + m[k+1][j] + ... }
  - Message passing propagates structural information
  - Split scoring: for (i,j), score each k using parent + child embeddings

Key Difference from Pointer Network:
  - Pointer Net: sequential encoder (reads dims left-to-right)
  - GNN: graph encoder (models sub-problem DEPENDENCIES directly)

Implemented in pure PyTorch (no torch_geometric dependency).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

NODE_FEAT_DIM = 10  # features per sub-chain node


# ─── Node Feature Extraction ─────────────────────────────────────────

def make_node_map(n):
    """
    Create mapping from sub-chain (i,j) to linear node index.
    Ordered by: all leaves first, then length-2, length-3, etc.
    This ordering helps bottom-up message passing.

    Args:
        n: number of matrices

    Returns:
        node_map: dict {(i,j): index}
        num_nodes: total number of nodes = n*(n+1)/2
    """
    node_map = {}
    idx = 0
    # Group by sub-chain length for natural bottom-up order
    for length in range(1, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            node_map[(i, j)] = idx
            idx += 1
    return node_map, idx


def extract_node_features(dims):
    """
    Extract features for every sub-chain (i,j) in the MCM problem.

    Args:
        dims: list of n+1 dimension values [d0, d1, ..., dn]

    Returns:
        features: (num_nodes, NODE_FEAT_DIM) numpy array
        node_map: dict mapping (i,j) -> node_index
        n: number of matrices
    """
    n = len(dims) - 1
    arr = np.array(dims, dtype=np.float64)
    log_arr = np.log1p(arr)

    node_map, num_nodes = make_node_map(n)
    features = np.zeros((num_nodes, NODE_FEAT_DIM), dtype=np.float32)

    for (i, j), nidx in node_map.items():
        sub_dims = arr[i - 1: j + 1]       # d[i-1] through d[j]
        sub_log = log_arr[i - 1: j + 1]
        L = j - i + 1

        features[nidx, 0] = log_arr[i - 1]                              # left boundary
        features[nidx, 1] = log_arr[j]                                   # right boundary
        features[nidx, 2] = math.log1p(arr[i - 1] * arr[j])             # boundary product
        features[nidx, 3] = L / max(n, 1)                                # normalized length
        features[nidx, 4] = sub_log.mean()                               # mean log-dim
        features[nidx, 5] = sub_log.std() if len(sub_log) > 1 else 0.0  # std log-dim
        features[nidx, 6] = math.log1p(sub_dims.min())                   # log min dim
        features[nidx, 7] = math.log1p(sub_dims.max())                   # log max dim
        features[nidx, 8] = np.argmin(sub_dims) / max(len(sub_dims) - 1, 1)  # min position
        features[nidx, 9] = 1.0 if L == 1 else 0.0                       # is_leaf flag

    return features, node_map, n


def build_edges(node_map, n):
    """
    Build edge list connecting each sub-chain to ALL its DP children.

    For each sub-chain (i,j) and each possible split k in [i, j-1]:
      - (i,j) <-> (i,k)     [left child]
      - (i,j) <-> (k+1,j)   [right child]

    Also adds same-length neighbor edges:
      - (i,j) <-> (i+1,j+1)  [adjacent sub-chain of same length]

    This gives the GNN direct access to the DP recurrence structure.

    Returns:
        edge_index: (2, num_edges) numpy array [sources, destinations]
    """
    sources = []
    destinations = []

    # DP child edges: (i,j) <-> (i,k) and (i,j) <-> (k+1,j)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            parent = node_map[(i, j)]

            for k in range(i, j):
                left_child = node_map[(i, k)]
                right_child = node_map[(k + 1, j)]

                # Bidirectional: parent <-> left_child
                sources.append(parent)
                destinations.append(left_child)
                sources.append(left_child)
                destinations.append(parent)

                # Bidirectional: parent <-> right_child
                sources.append(parent)
                destinations.append(right_child)
                sources.append(right_child)
                destinations.append(parent)

    # Same-length neighbor edges: (i,j) <-> (i+1,j+1)
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            if j + 1 <= n:
                n1 = node_map[(i, j)]
                n2 = node_map[(i + 1, j + 1)]
                sources.append(n1)
                destinations.append(n2)
                sources.append(n2)
                destinations.append(n1)

    if len(sources) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    return np.array([sources, destinations], dtype=np.int64)


def build_split_index_maps(node_map, n):
    """
    Precompute index maps for vectorized split scoring.

    For each sub-chain length L, returns arrays of node indices
    for parent, left-child, and right-child — enabling batch scoring.

    Returns:
        split_indices: dict {L: (parent_indices, left_indices, right_indices)}
            parent_indices: (num_sub,) node indices of parent sub-chains
            left_indices:   (num_sub, L-1) node indices of left children
            right_indices:  (num_sub, L-1) node indices of right children
    """
    split_indices = {}

    for L in range(2, n + 1):
        num_sub = n - L + 1
        num_cand = L - 1

        parents = np.zeros(num_sub, dtype=np.int64)
        lefts = np.zeros((num_sub, num_cand), dtype=np.int64)
        rights = np.zeros((num_sub, num_cand), dtype=np.int64)

        for s in range(num_sub):
            i = s + 1
            j = s + L
            parents[s] = node_map[(i, j)]

            for c, k in enumerate(range(i, j)):
                lefts[s, c] = node_map[(i, k)]
                rights[s, c] = node_map[(k + 1, j)]

        split_indices[L] = (parents, lefts, rights)

    return split_indices


# ─── GNN Layers ──────────────────────────────────────────────────────

class MessagePassingLayer(nn.Module):
    """
    Single GNN message-passing layer with gated aggregation.

    For each node, collects messages from neighbors, aggregates them,
    and updates the node embedding with a gated residual connection.
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.message_fn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        self.update_fn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, edge_index, num_nodes):
        """
        Args:
            x: (total_nodes, d_model) node embeddings
            edge_index: (2, num_edges) [source, destination] as LongTensor
            num_nodes: total number of nodes
        """
        src_idx, dst_idx = edge_index[0], edge_index[1]

        # Compute messages from source→destination
        src_feat = x[src_idx]
        dst_feat = x[dst_idx]
        messages = self.message_fn(torch.cat([src_feat, dst_feat], dim=-1))

        # Mean-aggregate messages per destination node
        aggregated = torch.zeros(num_nodes, self.d_model, device=x.device, dtype=messages.dtype)
        count = torch.zeros(num_nodes, 1, device=x.device, dtype=messages.dtype)
        aggregated.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(messages), messages)
        count.scatter_add_(0, dst_idx.unsqueeze(-1),
                           torch.ones(dst_idx.size(0), 1, device=x.device, dtype=messages.dtype))
        count = count.clamp(min=1)
        aggregated = aggregated / count

        # Gated residual update
        gate_input = torch.cat([x, aggregated], dim=-1)
        gate_val = self.gate(gate_input)
        update = self.update_fn(gate_input)
        x_new = x + gate_val * update

        return self.norm(x_new)


class SplitScorer(nn.Module):
    """
    Scores candidate split points for a sub-chain.

    For sub-chain (i,j) split at k:
      score(k) = MLP( embed(i,j) || embed(i,k) || embed(k+1,j) )
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, parent_emb, left_emb, right_emb):
        """
        Vectorized scoring.

        Args:
            parent_emb: (N, d_model)
            left_emb:   (N, d_model)
            right_emb:  (N, d_model)

        Returns:
            scores: (N,) scalar score per candidate
        """
        combined = torch.cat([parent_emb, left_emb, right_emb], dim=-1)
        return self.scorer(combined).squeeze(-1)


# ─── Main Model ──────────────────────────────────────────────────────

class GraphMCMNet(nn.Module):
    """
    Graph Neural Network for MCM split prediction.

    Architecture:
      1. Node embedding: Linear(10 -> d_model)
      2. K rounds of message passing over the sub-problem graph
      3. Split scoring: for each (i,j), score all candidate splits
         using parent + child pair embeddings (vectorized)
      4. Auxiliary cost head: predict log1p(cost) from root node

    Output format matches PointerMCMNet for fair comparison:
      - split_logits: dict {L: (batch, num_sub, L-1)}
      - split_valid:  dict {L: (batch, num_sub)}
      - aux_cost:     (batch, 1)
    """

    def __init__(
        self,
        node_feat_dim=NODE_FEAT_DIM,
        d_model=128,
        num_layers=6,
        dropout=0.1,
        max_n=50,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_n = max_n
        self.num_layers = num_layers

        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        self.gnn_layers = nn.ModuleList([
            MessagePassingLayer(d_model, dropout)
            for _ in range(num_layers)
        ])

        self.split_scorer = SplitScorer(d_model, dropout)

        self.cost_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, node_features, edge_index, batch_info):
        """
        Vectorized forward pass on a batched graph.

        Args:
            node_features: (total_nodes, node_feat_dim)
            edge_index:    (2, total_edges) LongTensor
            batch_info: dict with:
                - 'num_nodes': int
                - 'batch_size': int
                - 'actual_lengths': (batch_size,) LongTensor
                - 'split_parent_idx': dict {L: (batch, num_sub) LongTensor}
                - 'split_left_idx':   dict {L: (batch, num_sub, L-1) LongTensor}
                - 'split_right_idx':  dict {L: (batch, num_sub, L-1) LongTensor}
                - 'split_valid':      dict {L: (batch, num_sub) BoolTensor}
                - 'root_indices':     (batch_size,) LongTensor
        """
        device = node_features.device
        batch_size = batch_info['batch_size']
        actual_lengths = batch_info['actual_lengths']
        max_n_batch = actual_lengths.max().item()

        # 1. Embed nodes
        x = self.node_embed(node_features)

        # 2. Message passing
        for layer in self.gnn_layers:
            x = layer(x, edge_index, batch_info['num_nodes'])

        # 3. Score splits — vectorized by sub-chain length L
        split_logits = {}
        split_valid_out = {}

        for L in range(2, max_n_batch + 1):
            if L not in batch_info['split_parent_idx']:
                continue

            parent_idx = batch_info['split_parent_idx'][L]    # (batch, num_sub)
            left_idx = batch_info['split_left_idx'][L]        # (batch, num_sub, L-1)
            right_idx = batch_info['split_right_idx'][L]      # (batch, num_sub, L-1)
            valid = batch_info['split_valid'][L]               # (batch, num_sub)

            num_sub = parent_idx.size(1)
            num_cand = L - 1

            # Gather embeddings — flatten for batch scoring
            # Parent: expand to match candidates
            p_idx_flat = parent_idx.unsqueeze(-1).expand(-1, -1, num_cand)  # (B, S, C)
            p_idx_flat = p_idx_flat.reshape(-1)                              # (B*S*C)
            l_idx_flat = left_idx.reshape(-1)                                # (B*S*C)
            r_idx_flat = right_idx.reshape(-1)                               # (B*S*C)

            parent_embs = x[p_idx_flat]   # (B*S*C, d_model)
            left_embs = x[l_idx_flat]     # (B*S*C, d_model)
            right_embs = x[r_idx_flat]    # (B*S*C, d_model)

            # Score all candidates at once
            scores_flat = self.split_scorer(parent_embs, left_embs, right_embs)
            scores = scores_flat.reshape(batch_size, num_sub, num_cand)

            # Mask invalid sub-chains
            scores = scores.masked_fill(~valid.unsqueeze(-1), float('-inf'))

            split_logits[L] = scores
            split_valid_out[L] = valid

        # 4. Cost prediction from root nodes
        root_embs = x[batch_info['root_indices']]
        aux_cost = self.cost_head(root_embs)

        return split_logits, split_valid_out, aux_cost

    def predict(self, node_features, edge_index, batch_info):
        """
        Inference: predict split table and compute cost.

        Returns:
            predicted_splits: list of dicts {(i,j): k} per sample
            aux_cost: (batch, 1) cost predictions
        """
        self.eval()
        with torch.no_grad():
            split_logits, split_valid, aux_cost = self.forward(
                node_features, edge_index, batch_info
            )

        batch_size = batch_info['batch_size']
        actual_lengths = batch_info['actual_lengths']
        predicted_splits = [{} for _ in range(batch_size)]

        # Fill trivial splits (single matrix and length-2)
        for b in range(batch_size):
            n = actual_lengths[b].item()
            for i in range(1, n + 1):
                predicted_splits[b][(i, i)] = i
            for i in range(1, n):
                predicted_splits[b][(i, i + 1)] = i  # only one choice

        # Fill predicted splits from logits
        for L, logits in split_logits.items():
            valid = split_valid[L]
            preds = logits.argmax(dim=-1)   # (batch, num_sub)

            for b in range(batch_size):
                n = actual_lengths[b].item()
                num_sub = max(0, n - L + 1)
                for s in range(num_sub):
                    if valid[b, s]:
                        i = s + 1
                        j = s + L
                        k = i + preds[b, s].item()
                        predicted_splits[b][(i, j)] = k

        return predicted_splits, aux_cost


# ─── Utility: Parenthesization ───────────────────────────────────────

def reconstruct_parenthesization(splits, n, matrix_names=None):
    """
    Convert a split table to human-readable parenthesization.

    Args:
        splits: dict {(i,j): k} — split points
        n: number of matrices
        matrix_names: optional list of names, default A1..An

    Returns:
        string like "((A1 × (A2 × A3)) × A4)"
    """
    if matrix_names is None:
        matrix_names = [f"A{i}" for i in range(1, n + 1)]

    def _build(i, j):
        if i == j:
            return matrix_names[i - 1]
        k = splits.get((i, j), (i + j) // 2)
        left = _build(i, k)
        right = _build(k + 1, j)
        return f"({left} × {right})"

    return _build(1, n)

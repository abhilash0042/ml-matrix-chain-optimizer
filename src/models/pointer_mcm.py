"""
Pointer Network for Matrix Chain Multiplication.
==================================================
Predicts optimal split points for all sub-chains, then computes
exact cost from the predicted splits.

Architecture:
  Encoder:  Linear(8 → d_model) + PositionalEncoding + TransformerEncoder
  Decoder:  For each sub-chain (i,j), Bahdanau attention over candidate
            split positions → probability distribution P(split=k | i,j)
  Aux Head: Mean-pooled encoder output → MLP → log1p(cost) prediction

Indexing Convention:
  - Dimensions: d[0], d[1], ..., d[n]  (0-indexed, n+1 values)
  - Matrices: M1, M2, ..., Mn  (1-indexed, n matrices)
  - DP table: s[i][j] for 1 <= i < j <= n
  - Encoder hidden: h[0], h[1], ..., h[n]  (0-indexed, matching dims)
  - For sub-chain (i,j): query uses h[i-1] and h[j],
    candidates are h[i], h[i+1], ..., h[j-1]
  - Target: s[i][j] - i  (relative index within candidates)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (reused from HybridTransformer)."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PointerMCMNet(nn.Module):
    """
    Pointer Network for MCM split prediction.

    Predicts the optimal split point for every sub-chain (i,j),
    enabling exact cost computation from the predicted parenthesization.
    """

    def __init__(
        self,
        input_dim=8,        # per-position features
        d_model=128,        # encoder hidden dimension
        nhead=8,            # attention heads
        num_layers=6,       # transformer encoder layers
        dim_feedforward=512,
        dropout=0.1,
        max_len=51,         # max sequence length (n=50 → 51 dims)
        max_n=50,           # max number of matrices
        attention_dim=128,  # Bahdanau attention hidden dim
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.max_n = max_n
        self.attention_dim = attention_dim

        # === ENCODER ===
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # === POINTER DECODER (Bahdanau Attention) ===
        # Query MLP: takes [h_start, h_end, length_embedding] → query vector
        self.length_embedding = nn.Embedding(max_n + 1, 32)  # embed sub-chain length
        self.query_mlp = nn.Sequential(
            nn.Linear(d_model * 2 + 32, attention_dim),
            nn.SiLU(),
        )

        # Bahdanau attention components
        # score(k) = V^T * tanh(W1*query + W2*h_k)
        self.W1 = nn.Linear(attention_dim, attention_dim, bias=False)
        self.W2 = nn.Linear(d_model, attention_dim, bias=False)
        self.V = nn.Linear(attention_dim, 1, bias=False)

        # === AUXILIARY COST HEAD ===
        self.cost_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def encode(self, seq_features, padding_mask):
        """
        Encode the dimension sequence.

        Args:
            seq_features: (batch, max_len, 8) per-position features
            padding_mask: (batch, max_len) True for padded positions

        Returns:
            h: (batch, max_len, d_model) encoder hidden states
        """
        x = self.embedding(seq_features)
        x = self.pos_encoder(x)
        h = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        return h

    def predict_splits_for_length(self, h, L, actual_lengths, device):
        """
        Predict split points for all sub-chains of a given length.

        Args:
            h: (batch, max_len, d_model) encoder hidden states
            L: sub-chain length (number of matrices in sub-chain)
            actual_lengths: (batch,) number of matrices per sample
            device: torch device

        Returns:
            logits: (batch, num_sub, L-1) split logits
            valid_mask: (batch, num_sub) which sub-chains are valid
        """
        batch_size = h.size(0)
        num_sub = self.max_n - L + 1  # max possible sub-chains of length L

        if num_sub <= 0:
            return None, None

        # Sub-chain indices (1-indexed matrix indices)
        # Sub-chain s: matrices (s+1) to (s+L), for s = 0, 1, ..., num_sub-1
        # Start dim position (0-indexed): s      = (s+1) - 1
        # End dim position (0-indexed):   s + L

        start_pos = torch.arange(0, num_sub, device=device)        # [0, 1, ..., num_sub-1]
        end_pos = start_pos + L                                      # [L, L+1, ..., L+num_sub-1]

        # Gather start/end hidden states
        h_start = h[:, start_pos, :]   # (batch, num_sub, d_model)
        h_end = h[:, end_pos, :]       # (batch, num_sub, d_model)

        # Length embedding (same for all sub-chains of this length)
        len_idx = torch.tensor([L], device=device)
        len_emb = self.length_embedding(len_idx)                     # (1, 32)
        len_emb = len_emb.unsqueeze(0).expand(batch_size, num_sub, -1)  # (batch, num_sub, 32)

        # Query
        query_input = torch.cat([h_start, h_end, len_emb], dim=-1)  # (batch, num_sub, 2*d_model+32)
        query = self.query_mlp(query_input)                          # (batch, num_sub, attention_dim)

        # Candidate hidden states
        # For sub-chain s, matrix index i = s+1, j = s+L
        # Valid split points: k = i, i+1, ..., j-1 = s+1, s+2, ..., s+L-1
        # In encoder output, split k is at position k (0-indexed dim position)
        # Number of candidates = L - 1
        candidate_offsets = torch.arange(1, L, device=device)        # [1, 2, ..., L-1]
        candidate_positions = start_pos.unsqueeze(1) + candidate_offsets.unsqueeze(0)
        # Shape: (num_sub, L-1), values are dim positions of candidate splits

        # Gather candidate hidden states: (batch, num_sub, L-1, d_model)
        cp = candidate_positions.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_sub, L-1)
        cp_flat = cp.reshape(batch_size, -1)                              # (batch, num_sub*(L-1))
        h_cand_flat = torch.gather(
            h, 1, cp_flat.unsqueeze(-1).expand(-1, -1, self.d_model)
        )  # (batch, num_sub*(L-1), d_model)
        h_candidates = h_cand_flat.reshape(batch_size, num_sub, L - 1, self.d_model)

        # Bahdanau attention
        query_proj = self.W1(query).unsqueeze(2)                     # (batch, num_sub, 1, attn_dim)
        key_proj = self.W2(h_candidates)                             # (batch, num_sub, L-1, attn_dim)
        scores = self.V(torch.tanh(query_proj + key_proj)).squeeze(-1)  # (batch, num_sub, L-1)

        # Valid mask: sub-chain (s+1, s+L) is valid only if s+L <= n_i
        # n_i = actual_lengths[i] (number of matrices for sample i)
        # s + L <= n_i  →  s <= n_i - L
        max_valid_s = (actual_lengths - L).clamp(min=-1)             # (batch,)
        s_indices = torch.arange(num_sub, device=device).unsqueeze(0)  # (1, num_sub)
        valid_mask = s_indices <= max_valid_s.unsqueeze(1)           # (batch, num_sub)

        # Mask out invalid sub-chains in scores
        invalid_mask = ~valid_mask
        scores = scores.masked_fill(invalid_mask.unsqueeze(-1), float('-inf'))

        return scores, valid_mask

    def forward(self, seq_features, padding_mask, actual_lengths):
        """
        Full forward pass: encode + predict all splits + auxiliary cost.

        Args:
            seq_features: (batch, max_len, 8)
            padding_mask: (batch, max_len) True for padded
            actual_lengths: (batch,) number of matrices per sample

        Returns:
            split_logits: dict mapping length L → (batch, num_sub, L-1) logits
            split_valid: dict mapping length L → (batch, num_sub) validity mask
            aux_cost: (batch, 1) auxiliary cost prediction (log1p space)
        """
        device = seq_features.device
        h = self.encode(seq_features, padding_mask)

        # Predict splits for all sub-chain lengths
        # Skip L=2: only 1 candidate, trivially correct (split at i)
        max_n_batch = actual_lengths.max().item()
        split_logits = {}
        split_valid = {}

        for L in range(3, max_n_batch + 1):
            logits, valid = self.predict_splits_for_length(
                h, L, actual_lengths, device
            )
            if logits is not None:
                split_logits[L] = logits
                split_valid[L] = valid

        # Auxiliary cost prediction (masked mean pooling)
        weights = (~padding_mask).float().unsqueeze(-1)
        pooled = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        aux_cost = self.cost_head(pooled)

        return split_logits, split_valid, aux_cost

    def predict(self, seq_features, padding_mask, actual_lengths):
        """
        Inference: predict split table and compute cost.

        Returns:
            predicted_splits: list of dicts {(i,j): k} per sample
            predicted_costs: list of costs computed from predicted splits
            aux_costs: (batch,) auxiliary cost predictions
        """
        self.eval()
        with torch.no_grad():
            split_logits, split_valid, aux_cost = self.forward(
                seq_features, padding_mask, actual_lengths
            )

        batch_size = seq_features.size(0)
        predicted_splits = [{} for _ in range(batch_size)]

        # Fill trivial splits (length-2 sub-chains: only one choice)
        for b in range(batch_size):
            n = actual_lengths[b].item()
            for i in range(1, n + 1):
                predicted_splits[b][(i, i)] = i  # single matrix, no split needed

            # Length 2: only one candidate, split at i
            for i in range(1, n):
                predicted_splits[b][(i, i + 1)] = i

        # Fill predicted splits from logits
        for L, logits in split_logits.items():
            valid = split_valid[L]
            preds = logits.argmax(dim=-1)  # (batch, num_sub) — relative indices

            for b in range(batch_size):
                n = actual_lengths[b].item()
                num_sub = max(0, n - L + 1)
                for s in range(num_sub):
                    if valid[b, s]:
                        i = s + 1          # 1-indexed matrix start
                        j = s + L          # 1-indexed matrix end
                        k = i + preds[b, s].item()  # absolute split point
                        predicted_splits[b][(i, j)] = k

        return predicted_splits, aux_cost


def compute_cost_from_splits(dims, splits):
    """
    Compute multiplication cost from a predicted split table.

    Args:
        dims: list of n+1 dimension values
        splits: dict {(i,j): k} — predicted split points (1-indexed)

    Returns:
        cost: total multiplication cost under this parenthesization
    """
    n = len(dims) - 1
    memo = {}

    def _cost(i, j):
        if i == j:
            return 0
        if (i, j) in memo:
            return memo[(i, j)]

        k = splits.get((i, j))
        if k is None:
            # Fallback: split in the middle
            k = (i + j) // 2

        left = _cost(i, k)
        right = _cost(k + 1, j)
        combine = dims[i - 1] * dims[k] * dims[j]

        total = left + right + combine
        memo[(i, j)] = total
        return total

    return _cost(1, n)

"""
Transformer v2 for Matrix Chain Multiplication Split Prediction.
=================================================================
A pure Transformer encoder-decoder that predicts optimal split points
for all sub-chains, then computes exact cost from the predicted splits.

Architecture:
  Encoder:  Linear(8 → d_model) + PositionalEncoding + TransformerEncoder
  Decoder:  For each sub-chain (i,j), multi-head cross-attention over
            candidate split positions → probability distribution P(split=k | i,j)
  Aux Head: Mean-pooled encoder output → MLP → log1p(cost) prediction

Key Difference from PointerMCMNet:
  - Pointer Network uses Bahdanau (additive) attention: V^T * tanh(W1*q + W2*k)
  - This model uses scaled dot-product multi-head cross-attention: softmax(QK^T/√d)V
  - Adds self-attention between queries of the same sub-chain length
  - Multi-head attention captures richer structural patterns

Indexing Convention (same as PointerMCMNet):
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
    """Sinusoidal positional encoding."""

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


class CrossAttentionSplitDecoder(nn.Module):
    """
    Cross-attention decoder for split point prediction.
    
    For each sub-chain (i,j), generates a query from boundary hidden states,
    then attends over candidate split positions using multi-head cross-attention.
    
    This is the KEY difference from the Pointer Network's Bahdanau decoder.
    Instead of:  score = V^T * tanh(W1*query + W2*key)  (single-head additive)
    We use:      score = softmax(QK^T / √d_k)           (multi-head dot-product)
    """

    def __init__(self, d_model=128, nhead=8, num_decoder_layers=1,
                 dim_feedforward=512, dropout=0.1, max_n=50):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.max_n = max_n

        # Sub-chain length embedding
        self.length_embedding = nn.Embedding(max_n + 1, 32)

        # Query generation: [h_start, h_end, length_emb] → query
        self.query_proj = nn.Sequential(
            nn.Linear(d_model * 2 + 32, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # Multi-head cross-attention layers
        # Each layer: self-attn over queries + cross-attn over candidates + FFN
        self.decoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(
                    d_model, nhead, dropout=dropout, batch_first=True
                ),
                'self_attn_norm': nn.LayerNorm(d_model),
                'cross_attn': nn.MultiheadAttention(
                    d_model, nhead, dropout=dropout, batch_first=True
                ),
                'cross_attn_norm': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout),
                ),
                'ffn_norm': nn.LayerNorm(d_model),
            })
            for _ in range(num_decoder_layers)
        ])

        # Final scoring head: project each candidate to a scalar logit
        self.score_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, h, L, actual_lengths, device):
        """
        Predict split points for all sub-chains of length L.

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

        # ─── 1. Generate Queries ────────────────────────────────────────

        start_pos = torch.arange(0, num_sub, device=device)
        end_pos = start_pos + L

        h_start = h[:, start_pos, :]   # (batch, num_sub, d_model)
        h_end = h[:, end_pos, :]       # (batch, num_sub, d_model)

        len_idx = torch.tensor([L], device=device)
        len_emb = self.length_embedding(len_idx)  # (1, 32)
        len_emb = len_emb.unsqueeze(0).expand(batch_size, num_sub, -1)

        query_input = torch.cat([h_start, h_end, len_emb], dim=-1)
        queries = self.query_proj(query_input)  # (batch, num_sub, d_model)

        # ─── 2. Prepare Candidates (Keys/Values) ────────────────────────
        # For a sub-chain of length L, there are L-1 possible split points.
        # Candidates are the encoder hidden states of the matrices in the sub-chain.
        
        # USE UNFOLD: This is a high-speed, zero-copy way to get sliding windows.
        # Matrix features start at index 1 (index 0 is <BOS> token).
        h_matrices = h[:, 1:, :]  # Shape: (batch, seq_len-1, d_model)
        
        # Unfold to get sliding windows of size L-1
        h_candidates = h_matrices.unfold(1, L - 1, 1).permute(0, 1, 3, 2)
        
        # Keep exactly 'num_sub' windows and make contiguous for reshaping
        h_candidates = h_candidates[:, :num_sub, :, :].contiguous()

        # ─── 3. Valid Mask ──────────────────────────────────────────────
        # Only sub-chains that fit within the actual chain length are valid.
        max_valid_s = (actual_lengths - L).clamp(min=-1)
        s_indices = torch.arange(num_sub, device=device).unsqueeze(0)
        valid_mask = s_indices <= max_valid_s.unsqueeze(1)  # (batch, num_sub)

        # ─── 4. Cross-Attention Decoder Layers ──────────────────────────

        # Reshape candidates for per-sub-chain cross-attention:
        # candidates: (batch * num_sub, L-1, d_model) — keys/values
        kv = h_candidates.reshape(batch_size * num_sub, L - 1, self.d_model)

        q_seq = queries  # (batch, num_sub, d_model) — for self-attention

        # Self-attention mask: invalid sub-chains should not participate
        invalid_mask = ~valid_mask  # (batch, num_sub)

        # FIX: Prevent NaN from all-masked self-attention.
        # When a batch element has n < L, ALL sub-chains are invalid,
        # making key_padding_mask all-True. softmax([-inf,...]) = NaN.
        # Solution: unmask one dummy position for fully-masked elements.
        # This dummy position participates in attention but its output
        # is masked out in the final scores, so it doesn't affect the loss.
        safe_invalid_mask = invalid_mask.clone()
        all_masked = invalid_mask.all(dim=1)  # (batch,)
        if all_masked.any():
            safe_invalid_mask[all_masked, 0] = False

        for layer in self.decoder_layers:
            # Self-attention over queries (batch, num_sub, d_model)
            # This lets sub-chains share information about split patterns
            q_residual = q_seq
            q_normed = layer['self_attn_norm'](q_seq)
            q_self, _ = layer['self_attn'](
                q_normed, q_normed, q_normed,
                key_padding_mask=safe_invalid_mask
            )
            # Safety: replace any residual NaN with 0
            q_self = torch.nan_to_num(q_self, nan=0.0)
            q_seq = q_residual + q_self

            # Cross-attention: each query attends to its own candidates
            # Reshape for batched cross-attention
            q_flat = q_seq.reshape(batch_size * num_sub, 1, self.d_model)
            q_residual_flat = q_flat

            q_normed_flat = layer['cross_attn_norm'](q_flat)
            q_cross, _ = layer['cross_attn'](
                q_normed_flat, kv, kv
            )
            q_cross = torch.nan_to_num(q_cross, nan=0.0)
            q_flat = q_residual_flat + q_cross

            # FFN
            q_residual_flat = q_flat
            q_normed_flat = layer['ffn_norm'](q_flat)
            q_flat = q_residual_flat + layer['ffn'](q_normed_flat)

            # Reshape back
            q_seq = q_flat.reshape(batch_size, num_sub, self.d_model)

        # ─── 5. Score Candidates ────────────────────────────────────────

        # Use updated queries to score each candidate position
        # queries: (batch, num_sub, d_model)
        # candidates: (batch, num_sub, L-1, d_model)

        # Expand queries to match candidates
        q_expanded = q_seq.unsqueeze(2).expand(-1, -1, L - 1, -1)
        # Element-wise interaction + scoring
        interaction = q_expanded * h_candidates  # (batch, num_sub, L-1, d_model)
        scores = self.score_proj(interaction).squeeze(-1)  # (batch, num_sub, L-1)

        # Mask out invalid sub-chains
        scores = scores.masked_fill(invalid_mask.unsqueeze(-1), float('-inf'))

        return scores, valid_mask

    def forward_vectorized(self, queries, h_candidates, cand_mask, all_j, actual_lengths):
        """
        Vectorized forward pass for all sub-chains.
        """
        batch_size, num_queries, d_model = queries.shape
        max_cand = h_candidates.size(2)
        device = queries.device

        # 1. Valid mask for sub-chains (batch, num_queries)
        # sub-chain is valid if its end index all_j <= actual_length
        valid_mask = all_j.unsqueeze(0) <= actual_lengths.unsqueeze(1)
        invalid_mask = ~valid_mask

        # 2. Key Padding Mask for Cross-Attention (batch * num_queries, max_cand)
        # We must mask candidates that are padding (beyond L-1)
        # cand_mask: (num_queries, max_cand)
        cross_key_padding_mask = cand_mask.unsqueeze(0).expand(batch_size, -1, -1)
        cross_key_padding_mask = cross_key_padding_mask.reshape(batch_size * num_queries, max_cand)

        # 3. Key Padding Mask for Self-Attention (batch, num_queries)
        # Prevent NaN for all-masked self-attention
        safe_invalid_mask = invalid_mask.clone()
        all_masked = invalid_mask.all(dim=1)
        if all_masked.any():
            safe_invalid_mask[all_masked, 0] = False

        q_seq = queries
        kv = h_candidates.reshape(batch_size * num_queries, max_cand, d_model)

        for layer in self.decoder_layers:
            # Self-attention over queries
            q_residual = q_seq
            q_normed = layer['self_attn_norm'](q_seq)
            q_self, _ = layer['self_attn'](
                q_normed, q_normed, q_normed,
                key_padding_mask=safe_invalid_mask
            )
            q_self = torch.nan_to_num(q_self, nan=0.0)
            q_seq = q_residual + q_self

            # Cross-attention over candidates
            q_flat = q_seq.reshape(batch_size * num_queries, 1, d_model)
            q_residual_flat = q_flat
            q_normed_flat = layer['cross_attn_norm'](q_flat)
            
            q_cross, _ = layer['cross_attn'](
                q_normed_flat, kv, kv,
                key_padding_mask=cross_key_padding_mask
            )
            q_cross = torch.nan_to_num(q_cross, nan=0.0)
            q_flat = q_residual_flat + q_cross

            # FFN
            q_residual_flat = q_flat
            q_normed_flat = layer['ffn_norm'](q_flat)
            q_flat = q_residual_flat + layer['ffn'](q_normed_flat)
            q_seq = q_flat.reshape(batch_size, num_queries, d_model)

        # 4. Score Candidates
        q_expanded = q_seq.unsqueeze(2).expand(-1, -1, max_cand, -1)
        interaction = q_expanded * h_candidates
        scores = self.score_proj(interaction).squeeze(-1) # (batch, num_queries, max_cand)

        # Final masking
        # 1. Mask candidates beyond L-1
        scores = scores.masked_fill(cand_mask.unsqueeze(0), float('-inf'))
        # 2. Mask invalid sub-chains
        scores = scores.masked_fill(invalid_mask.unsqueeze(-1), float('-inf'))

        return scores, valid_mask


class TransformerMCMSplitNet(nn.Module):
    """
    Transformer v2 for MCM split prediction.

    Predicts the optimal split point for every sub-chain (i,j),
    enabling exact cost computation from the predicted parenthesization.

    Key differences from PointerMCMNet:
    1. Cross-attention decoder (multi-head dot-product) instead of Bahdanau (additive)
    2. Self-attention between queries of the same sub-chain length
    3. Deeper decoder (4 layers with residual connections + FFN)
    4. Multiplicative candidate scoring (query * candidate → score)
    """

    def __init__(
        self,
        input_dim=8,
        d_model=128,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=1,
        dim_feedforward=512,
        dropout=0.1,
        max_len=51,
        max_n=50,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.max_n = max_n

        # === ENCODER (same as PointerMCMNet) ===
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
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )

        # === CROSS-ATTENTION DECODER ===
        self.decoder = CrossAttentionSplitDecoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_n=max_n,
        )

        # === AUXILIARY COST HEAD (same as PointerMCMNet) ===
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
        h = self.encode(seq_features, padding_mask)
        batch_size, seq_len, d_model = h.shape
        device = h.device
        
        # 1. Generate all valid (s, s+L) pairs 
        max_n_batch = actual_lengths.max().item()
        
        all_i = []
        all_j = []
        all_L = []
        
        # Original loop structure but flattened:
        for L in range(3, max_n_batch + 1):
            num_sub = max_n_batch - L + 1
            for s in range(num_sub):
                all_i.append(s)
                all_j.append(s + L)
                all_L.append(L)
                
        if not all_i:
            return {}, {}, self.cost_head(self.encode(seq_features, padding_mask).sum(dim=1))
            
        all_i = torch.tensor(all_i, device=device)
        all_j = torch.tensor(all_j, device=device)
        L_vals = torch.tensor(all_L, device=device)
        num_queries = len(all_i)
        
        # 2. Construct Query Inputs
        h_start = h[:, all_i, :]
        h_end = h[:, all_j, :]
        len_emb = self.decoder.length_embedding(L_vals) 
        len_emb = len_emb.unsqueeze(0).expand(batch_size, -1, -1)
        query_input = torch.cat([h_start, h_end, len_emb], dim=-1)
        queries = self.decoder.query_proj(query_input)
        
        # 3. Construct Candidate Keys/Values (Padded)
        max_L = L_vals.max().item()
        max_cand = max_L - 1
        # Candidates start at i+1
        cand_indices = all_i.unsqueeze(1) + torch.arange(1, max_cand + 1, device=device).unsqueeze(0)
        # Mask out indices that are >= j (meaning they exceed the length L-1 candidates for this specific sub-chain)
        cand_mask = cand_indices >= all_j.unsqueeze(1)
        # Clamp indices to valid range for gather safety
        safe_indices = cand_indices.clamp(0, seq_len - 1)
        
        h_expanded = h.unsqueeze(1).expand(-1, num_queries, -1, -1)
        idx = safe_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, d_model)
        h_candidates = torch.gather(h_expanded, 2, idx)
        
        # 4. Run Vectorized Decoder
        logits_all, valid_mask_all = self.decoder.forward_vectorized(
            queries, h_candidates, cand_mask, all_j, actual_lengths
        )
        
        # 5. Map back to dictionary format
        split_logits = {}
        split_valid = {}
        
        # We need to map the flat results back to dictionaries of shape (batch, num_sub, L-1)
        start_idx = 0
        for L in range(3, max_n_batch + 1):
            num_sub = max_n_batch - L + 1
            end_idx = start_idx + num_sub
            
            logits_L = logits_all[:, start_idx:end_idx, :L-1]
            valid_L = valid_mask_all[:, start_idx:end_idx]
            
            split_logits[L] = logits_L
            split_valid[L] = valid_L
            
            start_idx = end_idx

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
            aux_cost: (batch, 1) auxiliary cost predictions
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
                predicted_splits[b][(i, i)] = i

            for i in range(1, n):
                predicted_splits[b][(i, i + 1)] = i

        # Fill predicted splits from logits
        for L, logits in split_logits.items():
            valid = split_valid[L]
            preds = logits.argmax(dim=-1)

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

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class HybridMCMTransformer(nn.Module):
    """
    Hybrid Transformer for MCM Cost Prediction.
    Combines sequence-level attention with engineered summary features.
    """
    def __init__(self, 
                 seq_input_dim=1, 
                 eng_feat_dim=177, 
                 d_model=128, 
                 nhead=8, 
                 num_layers=6, 
                 dim_feedforward=512, 
                 dropout=0.1):
        super().__init__()
        
        # 1. Sequence Branch (Transformer)
        self.embedding = nn.Linear(seq_input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 2. Engineered Features Branch (MLP)
        self.eng_mlp = nn.Sequential(
            nn.Linear(eng_feat_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU()
        )
        
        # 3. Hybrid Integration
        # Direct prediction head (fallback)
        self.regressor = nn.Sequential(
            nn.Linear(d_model + 128, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        
        # Residual correction head
        self.correction_head = nn.Sequential(
            nn.Linear(d_model + 128 + 1, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
            # Tanh removed to allow larger corrections
        )
        
        # Learnable scale for correction magnitude
        self.correction_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, seq, eng_feats, mask=None, greedy_baseline=None):
        # seq: (batch, seq_len, 1)
        # eng_feats: (batch, 177)
        
        # Process Sequence
        x_seq = self.embedding(seq)
        x_seq = self.pos_encoder(x_seq)
        x_seq = self.transformer_encoder(x_seq, src_key_padding_mask=mask)
        
        # Global Average Pooling for sequence (respecting mask)
        if mask is not None:
            weights = (~mask).float().unsqueeze(-1)
            x_seq = (x_seq * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        else:
            x_seq = x_seq.mean(dim=1)
            
        # Process Engineered Features
        x_eng = self.eng_mlp(eng_feats)
        
        # Combine
        combined = torch.cat([x_seq, x_eng], dim=1)
        
        if greedy_baseline is not None:
            # Residual correction: baseline + learned_correction
            correction_input = torch.cat([combined, greedy_baseline], dim=1)
            correction = self.correction_head(correction_input)
            return greedy_baseline + self.correction_scale * correction
        else:
            # Fallback: direct prediction
            return self.regressor(combined)

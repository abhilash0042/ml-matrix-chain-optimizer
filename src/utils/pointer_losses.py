"""
Loss functions for the Pointer Network.
========================================
Primary: Cross-entropy over split predictions (classification).
Auxiliary: Scale-aware MAPE on cost predictions (regression).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitPredictionLoss(nn.Module):
    """
    Cross-entropy loss for split point prediction.

    For each sub-chain (i,j), the model outputs logits over L-1 candidate
    split positions. The target is the index of the optimal split.

    Longer sub-chains are weighted more heavily because:
    1. They have more candidates (harder classification)
    2. Errors in longer sub-chains affect cost more
    """

    def __init__(self, length_weight=True):
        super().__init__()
        self.length_weight = length_weight

    def forward(self, split_logits, split_valid, split_targets, actual_lengths):
        """
        Args:
            split_logits: dict {L: (batch, num_sub, L-1)} from model
            split_valid: dict {L: (batch, num_sub)} validity mask
            split_targets: (batch, max_n, max_n) target relative indices
            actual_lengths: (batch,) number of matrices

        Returns:
            loss: weighted cross-entropy loss
            accuracy: fraction of correctly predicted splits
        """
        total_loss = 0.0
        total_weight = 0.0
        total_correct = 0
        total_count = 0

        for L, logits in split_logits.items():
            valid = split_valid[L]  # (batch, num_sub)
            batch_size, num_sub, num_candidates = logits.shape

            # Extract targets for this sub-chain length
            # Sub-chain s: matrices (s+1) to (s+L)
            # In split_targets: index [s][s+L-1] (0-indexed from 1-indexed i=s+1, j=s+L)
            targets = torch.zeros(batch_size, num_sub, dtype=torch.long,
                                  device=logits.device)

            for s in range(num_sub):
                i_idx = s      # 0-indexed row = i-1 = (s+1)-1 = s
                j_idx = s + L - 1  # 0-indexed col = j-1 = (s+L)-1 = s+L-1
                if j_idx < split_targets.size(2):
                    targets[:, s] = split_targets[:, i_idx, j_idx]

            # Flatten valid entries
            valid_flat = valid.reshape(-1)  # (batch * num_sub)
            logits_flat = logits.reshape(-1, num_candidates)  # (batch*num_sub, L-1)
            targets_flat = targets.reshape(-1)  # (batch * num_sub)

            # Filter to valid entries only
            valid_indices = valid_flat.nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                continue

            valid_logits = logits_flat[valid_indices]    # (V, L-1)
            valid_targets = targets_flat[valid_indices]  # (V,)

            # Clamp targets to valid range (safety)
            valid_targets = valid_targets.clamp(0, num_candidates - 1)

            # Cross-entropy loss
            ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction='mean')

            # Weight by sub-chain length
            w = float(L) if self.length_weight else 1.0
            total_loss += w * ce_loss * len(valid_indices)
            total_weight += w * len(valid_indices)

            # Accuracy
            preds = valid_logits.argmax(dim=-1)
            total_correct += (preds == valid_targets).sum().item()
            total_count += len(valid_indices)

        if total_weight == 0:
            return torch.tensor(0.0, device=split_targets.device, requires_grad=True), 0.0

        loss = total_loss / total_weight
        accuracy = total_correct / max(total_count, 1)

        return loss, accuracy


class PointerNetLoss(nn.Module):
    """
    Combined loss for the Pointer Network:
      - Primary: Split prediction cross-entropy (90%)
      - Auxiliary: Cost prediction LogCosh (10%)
    """

    def __init__(self, split_weight=0.9, cost_weight=0.1, length_weight=True):
        super().__init__()
        self.split_loss = SplitPredictionLoss(length_weight=length_weight)
        self.split_weight = split_weight
        self.cost_weight = cost_weight

    def forward(self, split_logits, split_valid, split_targets,
                actual_lengths, aux_cost_pred, cost_target):
        """
        Returns:
            total_loss, split_loss_value, cost_loss_value, split_accuracy
        """
        # Split prediction loss
        s_loss, s_acc = self.split_loss(
            split_logits, split_valid, split_targets, actual_lengths
        )

        # Cost prediction loss (LogCosh — smooth and outlier-robust)
        diff = aux_cost_pred - cost_target
        c_loss = torch.mean(torch.log(torch.cosh(diff + 1e-12)))

        # Combined
        total = self.split_weight * s_loss + self.cost_weight * c_loss

        return total, s_loss.item(), c_loss.item(), s_acc

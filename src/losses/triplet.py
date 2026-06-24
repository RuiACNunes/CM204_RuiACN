"""[CM-204] Triplet loss (margem default 0.2), consumindo os índices do miner.

    L(a, p, n) = max(0, d(a,p) - d(a,n) + margin)

com d = distância euclidiana sobre vetores L2-normalizados (equivalente a
distância de cosseno; SPEC §2.1, §6.4).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        anchor_idx: torch.Tensor,
        positive_idx: torch.Tensor,
        negative_idx: torch.Tensor,
    ) -> torch.Tensor:
        if anchor_idx.numel() == 0:
            return embeddings.sum() * 0.0

        a = embeddings[anchor_idx]
        p = embeddings[positive_idx]
        n = embeddings[negative_idx]

        d_ap = F.pairwise_distance(a, p, p=2)
        d_an = F.pairwise_distance(a, n, p=2)
        losses = F.relu(d_ap - d_an + self.margin)
        return losses.mean()

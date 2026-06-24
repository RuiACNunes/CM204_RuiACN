"""[CM-204] Cabeça de projeção treinada sobre os embeddings congelados do CLIP.

Arquitetura fixa pelo SPEC §6.1: Linear -> BatchNorm1d -> ReLU -> Linear ->
L2-normalize na saída. A normalização da saída é estrutural (não é flag): a
triplet loss com distância euclidiana sobre vetores normalizados equivale a
distância de cosseno, o regime estável para metric learning (SPEC §2.1).
"""
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)

"""[CM-204] SupCon (supervised contrastive) — ablação de perda futura.

Stub estrutural (SPEC §8): mesmo sampler e head da Layer 1, trocando apenas a
perda, para comparar triplet vs. supervised-contrastive. Implementação
opcional, fora do escopo do protótipo pré-26/06 (SPEC §11).
"""
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "SupConLoss é uma ablação futura (SPEC §8) — não implementada no protótipo atual."
        )

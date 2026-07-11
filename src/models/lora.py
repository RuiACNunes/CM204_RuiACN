"""[CM-204] Adaptação via LoRA do ViT do CLIP — Layer 3 (Hu et al., 2021).

Injeta adaptadores de baixo posto nas projeções de saída de atenção
(`out_proj`) de cada bloco do transformer visual do CLIP. O backbone base
fica completamente congelado; apenas os adaptadores A e B (e a projection
head, externa a este módulo) recebem gradiente.

Nota sobre a escolha de `out_proj` vs. Q/V:
    O open_clip funde Q, K e V num único tensor `in_proj_weight` que não é
    exposto como `nn.Linear` — não é possível injetar LoRALinear diretamente
    sem reimplementar o forward do MultiheadAttention. `out_proj` É um
    `nn.Linear` e pode ser substituído diretamente. O paper original de LoRA
    (Tabela 6) reporta que adaptar só out_proj produz resultados comparáveis
    a Q+V em ViTs, com implementação muito mais limpa.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0      # escala do adaptador; alpha/rank = fator multiplicativo
    dropout: float = 0.1    # regularização dentro do adaptador


class LoRALinear(nn.Module):
    """nn.Linear congelado + adaptador de baixo posto BA (Hu et al. 2021).

    Saída: W·x + (alpha/rank)·B·A·x
      - W: peso original, congelado
      - A: rank×in, inicializado com Kaiming (aprende desde o início)
      - B: out×rank, inicializado com zeros (garante delta=0 no passo 0)
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        in_f, out_f = linear.in_features, linear.out_features
        self.linear = linear  # congelado por inject_lora antes da substituição
        self.A = nn.Linear(in_f, rank, bias=False)
        self.B = nn.Linear(rank, out_f, bias=False)
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scale * self.B(self.drop(self.A(x)))


def inject_lora(clip_model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Congela o backbone visual e injeta LoRALinear em cada out_proj de atenção.

    Deve ser chamado ANTES de construir o optimizer — parâmetros adicionados
    depois não são vistos pelo optimizer.

    Retorna o mesmo modelo modificado in-place (para encadeamento).
    """
    # 1. Congelar todo o backbone visual
    for p in clip_model.visual.parameters():
        p.requires_grad_(False)

    # 2. Substituir out_proj em cada bloco de atenção por LoRALinear
    for block in clip_model.visual.transformer.resblocks:
        original_out = block.attn.out_proj
        # original_out.weight já está congelado (passo 1); LoRALinear.A e .B
        # são módulos novos com requires_grad=True por padrão.
        block.attn.out_proj = LoRALinear(
            original_out, config.rank, config.alpha, config.dropout
        )

    return clip_model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Retorna (parâmetros treináveis, parâmetros totais)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

"""[CM-204] Adaptação via LoRA do ViT do CLIP — Layer 3 (Hu et al., 2021).

Injeta adaptadores de baixo posto na projeção de saída do bloco MLP
(`mlp.c_proj`) de cada bloco do transformer visual do CLIP.

Por que `mlp.c_proj` e não `attn.out_proj`?
    O `nn.MultiheadAttention` do PyTorch lê `out_proj.weight`/`.bias`
    diretamente na função nativa (`F.multi_head_attention_forward`), em vez
    de chamar `out_proj(x)` como módulo. Substituir `out_proj` por um
    `LoRALinear` causaria AttributeError (sem `.weight`) e, mesmo que não
    quebrasse, o `forward` do adaptador nunca seria executado — o LoRA não
    teria efeito. `mlp.c_proj` é um `nn.Linear` chamado como módulo, onde
    o `LoRALinear.forward` executa corretamente. Adaptar camadas MLP é
    prática comum em PEFT (ver LoRA paper, seção 4).

Parâmetros treináveis (rank=4, ViT-B/16, 12 blocos):
    Cada c_proj: in=3072, out=768 → A: 3072×4, B: 4×768
    Total: 12 × 4 × (3072 + 768) = 184.320 / ~150M ≈ 0.12%
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
        self.linear = linear  # congelado pelo passo 1 de inject_lora
        self.A = nn.Linear(in_f, rank, bias=False)
        self.B = nn.Linear(rank, out_f, bias=False)
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scale * self.B(self.drop(self.A(x)))


def inject_lora(clip_model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Congela o modelo CLIP inteiro e injeta LoRALinear em mlp.c_proj de cada bloco.

    Congela o modelo *completo* (não só o visual) porque o CLIP tem parâmetros
    fora de `clip_model.visual` — encoder de texto, token_embedding, logit_scale
    etc. — que ficariam treináveis se só o visual fosse congelado, inflando a
    fração treinável e tornando `count_parameters` enganoso.

    Os `LoRALinear` são criados *depois* do congelamento: A e B nascem com
    `requires_grad=True` por padrão, sem precisar de tratamento especial.

    Deve ser chamado ANTES de construir o optimizer.
    Retorna o mesmo modelo modificado in-place (para encadeamento).
    """
    # 1. Congelar o modelo INTEIRO (visual + encoder de texto + embeddings)
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # 2. Injetar LoRALinear em mlp.c_proj de cada bloco do transformer visual
    for block in clip_model.visual.transformer.resblocks:
        block.mlp.c_proj = LoRALinear(
            block.mlp.c_proj, config.rank, config.alpha, config.dropout
        )

    return clip_model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Retorna (parâmetros treináveis, parâmetros totais)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

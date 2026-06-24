"""[CM-204] Adaptação via LoRA do ViT do CLIP — Layer 3 (resultado central planejado).

Stub estrutural (SPEC §8): aqui entrarão os adaptadores LoRA (Hu et al., 2021)
injetados nas projeções de atenção do ViT do CLIP, treinando apenas os
adaptadores + a projection head, com o restante do backbone congelado. Esta é
a única camada em que o backbone entra no grafo (forward sobre imagens, não
mais sobre embeddings cacheados) — exige GPU de verdade.

Fora do escopo do protótipo pré-26/06 (SPEC §11); entra após aprovação do tema.
"""
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 1.0
    target_modules: tuple = ("attn.in_proj", "attn.out_proj")


def inject_lora(clip_model, config: LoRAConfig):
    raise NotImplementedError(
        "Injeção de LoRA no ViT do CLIP é a Layer 3 (SPEC §8) — não implementada no protótipo atual."
    )

"""[VIAL-infra] Extração e cache dos embeddings congelados do CLIP.

Passa todas as imagens do CUB pelo backbone uma única vez e salva o resultado
em disco (SPEC §5.3). A partir daí, nada nas Layers 0-1 precisa de GPU: tudo
roda sobre o tensor cacheado.

O cache é sempre **raw** (não normalizado) — `ClipBackbone.encode` não aplica
L2-normalize. A normalização é decisão de cada consumidor (flag
`baseline_l2_normalize` no kNN da Layer 0; BatchNorm absorve a escala na
entrada da head da Layer 1, cuja saída é sempre L2-normalizada por
construção). Isso mantém o cache neutro: trocar a flag de ablação do
baseline não exige recachear nem retreinar a head.
"""
import os
from typing import List

import torch
from torch.utils.data import DataLoader

from src.data.cub import CUBDataset
from src.models.backbone import ClipBackbone


def cache_embeddings(
    dataset: CUBDataset,
    backbone: ClipBackbone,
    output_path: str,
    batch_size: int = 256,
    num_workers: int = 4,
) -> dict:
    """Itera sobre `dataset` completo, codifica com `backbone` e salva em
    `output_path` (.pt) um dict com:
      - embeddings: Tensor[N, D] float32
      - class_id:   Tensor[N] int64  (1-200)
      - image_id:   Tensor[N] int64
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    all_embeddings: List[torch.Tensor] = []
    all_class_ids: List[int] = []
    all_image_ids: List[int] = []

    for images, class_ids, image_ids in loader:
        feats = backbone.encode(images).cpu()
        all_embeddings.append(feats)
        all_class_ids.extend(class_ids.tolist() if torch.is_tensor(class_ids) else list(class_ids))
        all_image_ids.extend(image_ids.tolist() if torch.is_tensor(image_ids) else list(image_ids))

    payload = {
        "embeddings": torch.cat(all_embeddings, dim=0),
        "class_id": torch.tensor(all_class_ids, dtype=torch.int64),
        "image_id": torch.tensor(all_image_ids, dtype=torch.int64),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)
    return payload


def load_cached_embeddings(path: str) -> dict:
    return torch.load(path)

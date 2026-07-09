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

Formato em disco: NumPy `.npz` com três arrays — `embeddings`, `class_id`,
`image_id`. `load_cached_embeddings` devolve torch tensors para manter
compatibilidade com todos os consumidores sem alteração.
"""
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.cub import CUBDataset
from src.models.backbone import ClipBackbone


def cache_embeddings(
    dataset: CUBDataset,
    backbone: ClipBackbone,
    output_path: str,
    batch_size: int = 256,
    num_workers: int = 0,
) -> dict:
    """Itera sobre `dataset` completo, codifica com `backbone` e salva em
    `output_path` (.npz) com três arrays:
      - embeddings: float32 [N, D]
      - class_id:   int64   [N]   (1-200)
      - image_id:   int64   [N]

    Retorna dict com torch tensors (mesma estrutura de `load_cached_embeddings`).
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

    emb_tensor = torch.cat(all_embeddings, dim=0)
    cid_tensor = torch.tensor(all_class_ids, dtype=torch.int64)
    iid_tensor = torch.tensor(all_image_ids, dtype=torch.int64)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        embeddings=emb_tensor.numpy(),
        class_id=cid_tensor.numpy(),
        image_id=iid_tensor.numpy(),
    )

    return {"embeddings": emb_tensor, "class_id": cid_tensor, "image_id": iid_tensor}


def load_cached_embeddings(path: str) -> dict:
    """Carrega o cache `.npz` e devolve dict com torch tensors."""
    data = np.load(path)
    return {
        "embeddings": torch.from_numpy(data["embeddings"]),
        "class_id": torch.from_numpy(data["class_id"]),
        "image_id": torch.from_numpy(data["image_id"]),
    }

"""[CM-204] Estratégias de mineração de triplos.

Assinatura única `mine_triplets(embeddings, labels, strategy=...)`, já
projetada para acomodar a Layer 2 (SPEC §7) sem refatorar:
  - "batch_hard" (default da Layer 1): para cada âncora, o positivo mais
    distante e o negativo mais próximo dentro do batch (Hermans et al., 2017).
  - "batch_all": média da triplet loss sobre todos os triplos válidos do
    batch — ablação interna gratuita (triplet ingênua vs. minerada).
  - "semi_hard" / "distance_weighted": reservadas para a Layer 2 (ainda não
    implementadas).

Todas as funções operam sobre `embeddings` já L2-normalizados e devolvem
três tensores de índices (anchor_idx, positive_idx, negative_idx) relativos às
posições no batch.
"""
from typing import Tuple

import torch


def _pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.cdist(embeddings, embeddings, p=2)


def batch_hard_triplets(embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist = _pairwise_distances(embeddings)
    n = embeddings.size(0)

    same = labels.unsqueeze(1) == labels.unsqueeze(0)
    same.fill_diagonal_(False)
    diff = ~same
    diff.fill_diagonal_(False)

    anchors, positives, negatives = [], [], []
    for i in range(n):
        pos_idx = same[i].nonzero(as_tuple=True)[0]
        neg_idx = diff[i].nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        hardest_pos = pos_idx[torch.argmax(dist[i, pos_idx])]
        hardest_neg = neg_idx[torch.argmin(dist[i, neg_idx])]
        anchors.append(i)
        positives.append(hardest_pos.item())
        negatives.append(hardest_neg.item())

    device = embeddings.device
    return (
        torch.tensor(anchors, dtype=torch.long, device=device),
        torch.tensor(positives, dtype=torch.long, device=device),
        torch.tensor(negatives, dtype=torch.long, device=device),
    )


def batch_all_triplets(embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = embeddings.size(0)
    same = labels.unsqueeze(1) == labels.unsqueeze(0)
    same.fill_diagonal_(False)
    diff = ~same
    diff.fill_diagonal_(False)

    anchors, positives, negatives = [], [], []
    pos_pairs = same.nonzero(as_tuple=False)  # [num_pairs, 2] -> (anchor, positive)
    for a, p in pos_pairs.tolist():
        neg_candidates = diff[a].nonzero(as_tuple=True)[0].tolist()
        anchors.extend([a] * len(neg_candidates))
        positives.extend([p] * len(neg_candidates))
        negatives.extend(neg_candidates)

    device = embeddings.device
    return (
        torch.tensor(anchors, dtype=torch.long, device=device),
        torch.tensor(positives, dtype=torch.long, device=device),
        torch.tensor(negatives, dtype=torch.long, device=device),
    )


def mine_triplets(
    embeddings: torch.Tensor, labels: torch.Tensor, strategy: str = "batch_hard"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if strategy == "batch_hard":
        return batch_hard_triplets(embeddings, labels)
    elif strategy == "batch_all":
        return batch_all_triplets(embeddings, labels)
    elif strategy in ("semi_hard", "distance_weighted"):
        raise NotImplementedError(
            f"Estratégia '{strategy}' é reservada para a Layer 2 (SPEC §7) e ainda não foi implementada."
        )
    else:
        raise ValueError(f"Estratégia de mineração desconhecida: {strategy!r}")

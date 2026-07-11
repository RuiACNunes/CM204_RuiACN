"""[CM-204] Estratégias de mineração de triplos (SPEC §6.4, §7).

Assinatura única `mine_triplets(embeddings, labels, strategy, margin)`:
  - "batch_hard"        — Layer 1: positivo mais distante + negativo mais próximo (Hermans 2017)
  - "batch_all"         — Layer 1: todos os triplos válidos do batch (ablação interna)
  - "semi_hard"         — Layer 2: negativos dentro da margem mas além do positivo (FaceNet)
  - "distance_weighted" — Layer 2: amostragem inversamente proporcional à densidade (Wu 2017)

Todas as funções operam sobre `embeddings` L2-normalizados e devolvem três
tensores de índices (anchor, positive, negative) relativos ao batch.
"""
import math
from typing import Tuple

import torch


def _pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.cdist(embeddings, embeddings, p=2)


def batch_hard_triplets(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def batch_all_triplets(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = embeddings.size(0)
    same = labels.unsqueeze(1) == labels.unsqueeze(0)
    same.fill_diagonal_(False)
    diff = ~same
    diff.fill_diagonal_(False)

    anchors, positives, negatives = [], [], []
    pos_pairs = same.nonzero(as_tuple=False)
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


def semi_hard_triplets(
    embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FaceNet (Schroff et al. 2015): negativos que satisfazem d(a,p) < d(a,n) < d(a,p) + margin.

    Para cada par (âncora, positivo) no batch, seleciona aleatoriamente um
    negativo "semi-hard" — difícil o suficiente para gerar gradiente, mas não
    tão próximo que force o colapso do embedding.

    Política de fallback (RUNBOOK §A.2, opção ii): se nenhum negativo for
    semi-hard para aquele par, usa-se o negativo mais difícil disponível
    (batch_hard). Mais robusto que descartar o par (opção i), especialmente
    nas primeiras épocas quando a representação ainda não é boa.
    """
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
        for p in pos_idx:
            d_ap = dist[i, p].item()
            d_an = dist[i, neg_idx]
            # Semi-hard: d(a,p) < d(a,n) < d(a,p) + margin
            sh_mask = (d_an > d_ap) & (d_an < d_ap + margin)
            sh_neg = neg_idx[sh_mask]
            if sh_neg.numel() > 0:
                chosen = sh_neg[torch.randint(sh_neg.numel(), (1,)).item()]
            else:
                # Fallback: negativo mais difícil (batch_hard)
                chosen = neg_idx[torch.argmin(d_an)]
            anchors.append(i)
            positives.append(p.item())
            negatives.append(chosen.item())

    device = embeddings.device
    return (
        torch.tensor(anchors, dtype=torch.long, device=device),
        torch.tensor(positives, dtype=torch.long, device=device),
        torch.tensor(negatives, dtype=torch.long, device=device),
    )


def distance_weighted_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    cutoff: float = 0.5,
    nonzero_loss_cutoff: float = 1.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wu et al. 2017 (Sampling Matters in Deep Embedding Learning).

    Amostra negativos com probabilidade ∝ 1/q(d), onde q(d) é a densidade
    analítica de distâncias par-a-par numa hiperesfera unitária de dimensão D:

        q(d) ∝ d^(D-2) · (1 - d²/4)^((D-3)/2)    d ∈ [0, 2]

    Em alta dimensão, pares par-a-par se concentram em torno de d≈√2 (medida
    de concentração). Amostragem uniforme fica presa nessa faixa; a ponderação
    inversa espalha o espectro de dificuldades e reduz a variância do gradiente.

    cutoff: distância mínima para evitar explosão de peso perto de zero.
    nonzero_loss_cutoff: descarta negativos trivialmente fáceis além deste
        limiar — tipicamente ≈√2. Fallback para o negativo mais difícil se
        todos os negativos estiverem além do limiar.
    """
    n, D = embeddings.shape
    device = embeddings.device

    dist = _pairwise_distances(embeddings)

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

        d_neg = dist[i, neg_idx].clamp(min=cutoff, max=2.0 - 1e-6)

        # log w = -(D-2)·log(d) - (D-3)/2·log(1 - d²/4)
        log_w = -(D - 2) * torch.log(d_neg) - ((D - 3) / 2.0) * torch.log1p(
            -(d_neg ** 2) / 4.0
        )

        easy_mask = dist[i, neg_idx] > nonzero_loss_cutoff
        if easy_mask.all():
            # Todos os negativos são trivialmente fáceis; fallback para batch_hard
            chosen = neg_idx[torch.argmin(dist[i, neg_idx])].item()
        else:
            log_w = log_w.masked_fill(easy_mask, -1e9)
            w = torch.softmax(log_w.float(), dim=0)
            chosen = neg_idx[torch.multinomial(w, 1).item()].item()

        for p in pos_idx:
            anchors.append(i)
            positives.append(p.item())
            negatives.append(chosen)

    return (
        torch.tensor(anchors, dtype=torch.long, device=device),
        torch.tensor(positives, dtype=torch.long, device=device),
        torch.tensor(negatives, dtype=torch.long, device=device),
    )


def mine_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    strategy: str = "batch_hard",
    margin: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Despacha para a estratégia de mineração pedida.

    `margin` é usado apenas por `semi_hard` (define a janela de candidatos
    válidos); os demais estratégias ignoram-no.
    """
    if strategy == "batch_hard":
        return batch_hard_triplets(embeddings, labels)
    elif strategy == "batch_all":
        return batch_all_triplets(embeddings, labels)
    elif strategy == "semi_hard":
        return semi_hard_triplets(embeddings, labels, margin=margin)
    elif strategy == "distance_weighted":
        return distance_weighted_triplets(embeddings, labels)
    else:
        raise ValueError(f"Estratégia de mineração desconhecida: {strategy!r}")

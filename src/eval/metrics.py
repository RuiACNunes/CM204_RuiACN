"""[VIAL-infra herdado] Métricas de recuperação: Recall@k e mAP@R.

Ambas operam sobre o ranking leave-one-out produzido por
`src.eval.retrieval.leave_one_out_ranking` (já exclui a própria query).

Recall@k
--------
Para cada query i, hit_i = 1 se algum dos k primeiros itens do ranking tiver a
mesma `class_id` da query, senão 0.

    Recall@k = (1/N) * sum_i hit_i

Reportado para k em {1, 2, 4, 8} (SPEC §2).

mAP@R (Musgrave, Belongie & Lim, 2020 — "A Metric Learning Reality Check")
---------------------------------------------------------------------------
Para cada query i:
  - R_i = número de itens da mesma classe da query disponíveis na gallery
          (exclui a própria query).
  - Toma-se o top-R_i do ranking. Define-se, para a posição j (1-indexada)
    desse top-R_i:
        rel(j) = 1 se o item na posição j tem a mesma classe da query, senão 0
        P(j)   = precisão acumulada até a posição j = (1/j) * sum_{l=1}^{j} rel(l)

    AP@R_i = (1 / R_i) * sum_{j=1}^{R_i} rel(j) * P(j)

  - Queries com R_i = 0 (classe com uma única imagem no split de teste, sem
    nenhum positivo possível) são ignoradas na média, pois AP@R não é
    definida nesse caso.

    mAP@R = média de AP@R_i sobre todas as queries com R_i > 0.
"""
from typing import Dict, Iterable, List

import numpy as np


def recall_at_k(ranking: np.ndarray, labels: np.ndarray, ks: Iterable[int]) -> Dict[str, float]:
    labels = np.asarray(labels)
    results: Dict[str, float] = {}
    for k in ks:
        topk_labels = labels[ranking[:, :k]]
        hits = np.any(topk_labels == labels[:, None], axis=1)
        results[f"R@{k}"] = float(hits.mean())
    return results


def map_at_r(ranking: np.ndarray, labels: np.ndarray) -> float:
    n = ranking.shape[0]
    labels = np.asarray(labels)
    aps: List[float] = []
    for i in range(n):
        R = int(np.sum(labels == labels[i])) - 1  # exclui a própria query
        if R <= 0:
            continue
        top_r_labels = labels[ranking[i, :R]]
        relevant = (top_r_labels == labels[i]).astype(np.float64)
        cum_relevant = np.cumsum(relevant)
        precision_at_j = cum_relevant / (np.arange(R) + 1)
        ap = float(np.sum(relevant * precision_at_j) / R)
        aps.append(ap)
    return float(np.mean(aps)) if aps else float("nan")


def evaluate_retrieval(ranking: np.ndarray, labels: np.ndarray, recall_ks: Iterable[int]) -> Dict[str, float]:
    """Calcula Recall@k (para todo k em `recall_ks`) e mAP@R em um único dict,
    pronto para `src.utils.logging.log_experiment`.
    """
    metrics = recall_at_k(ranking, labels, recall_ks)
    metrics["mAP@R"] = map_at_r(ranking, labels)
    return metrics

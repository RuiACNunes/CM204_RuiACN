"""[VIAL-infra herdado] Índice FAISS + busca kNN exata, protocolo leave-one-out.

Protocolo (SPEC §5.4): cada embedding do split de teste é uma query; a gallery
é o restante do próprio split de teste (a query é excluída da sua própria
gallery). Não há gallery separada. Na escala deste estudo (~5.8k vetores de
teste), a busca exata é trivial; usa-se FAISS por consistência com a infra do
VIAL (uma busca exata em torch seria equivalente nesta escala).
"""
import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "faiss não encontrado. Instale `faiss-cpu` (ver requirements.txt)."
    ) from exc


def _build_index(embeddings: np.ndarray, metric: str = "cosine"):
    d = embeddings.shape[1]
    if metric == "cosine":
        # Para cosseno via FAISS, usa-se produto interno (IP) sobre vetores
        # L2-normalizados — equivalente a similaridade de cosseno.
        index = faiss.IndexFlatIP(d)
    elif metric == "euclidean":
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError(f"metric desconhecido: {metric!r} (use 'cosine' ou 'euclidean')")
    index.add(embeddings.astype(np.float32))
    return index


def leave_one_out_ranking(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Para cada vetor em `embeddings` (linha i), devolve o ranking completo dos
    demais N-1 vetores (índices, do mais ao menos similar/próximo), excluindo a
    própria query.

    Retorna: ndarray[N, N-1] de índices (em `embeddings`).
    """
    n = embeddings.shape[0]
    index = _build_index(embeddings, metric=metric)
    # busca todos os N vizinhos (inclui a própria query) para depois excluí-la.
    _, neighbor_idx = index.search(embeddings.astype(np.float32), n)

    ranking = np.empty((n, n - 1), dtype=np.int64)
    for i in range(n):
        row = neighbor_idx[i]
        row = row[row != i]
        ranking[i] = row[: n - 1]
    return ranking

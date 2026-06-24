"""Layer 0 — baseline congelado + kNN.

Carrega os embeddings cacheados (script 00), filtra o split de teste do
estudo (classes 101-200), roda a avaliação leave-one-out e loga o resultado
como regime `frozen_knn` — o piso de referência contra o qual toda a Layer 1
é medida (SPEC §5.5).

Uso:
    python scripts/01_baseline_knn.py [--config configs/default.yaml]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from src.cache.embeddings import load_cached_embeddings
from src.eval.metrics import evaluate_retrieval
from src.eval.retrieval import leave_one_out_ranking
from src.utils.logging import log_experiment
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    backbone_cfg = config["backbone"]
    embeddings_path = os.path.join(config["paths"]["embeddings_dir"], backbone_cfg["embedding_filename"])
    payload = load_cached_embeddings(embeddings_path)

    class_id = payload["class_id"].numpy()
    embeddings = payload["embeddings"].numpy()  # raw — não normalizado (ver src/cache/embeddings.py)

    split_cfg = config["split"]
    # metric_learning_split espera uma lista de objetos com atributo class_id;
    # aqui já temos os arrays diretamente, então filtramos pelo intervalo de classes.
    test_lo, test_hi = split_cfg["test_classes"]
    test_mask = (class_id >= test_lo) & (class_id <= test_hi)

    test_embeddings = embeddings[test_mask]
    test_labels = class_id[test_mask]
    print(f"Split de teste (classes {test_lo}-{test_hi}): {test_embeddings.shape[0]} imagens.")

    retrieval_cfg = config["retrieval"]
    baseline_l2_normalize = retrieval_cfg["baseline_l2_normalize"]
    if baseline_l2_normalize:
        norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        test_embeddings = test_embeddings / norms

    ranking = leave_one_out_ranking(test_embeddings, metric=retrieval_cfg["metric"])
    metrics = evaluate_retrieval(ranking, test_labels, retrieval_cfg["recall_ks"])
    print("Métricas (frozen_knn):", metrics)

    log_experiment(
        regime="frozen_knn",
        backbone=f"{backbone_cfg['model_name']}/{backbone_cfg['pretrained']}",
        loss=None,
        mining_strategy=None,
        hyperparams={"dim": test_embeddings.shape[1]},
        l2_normalize=baseline_l2_normalize,
        seed=config["seed"],
        metrics=metrics,
        csv_path=config["logging"]["experiments_csv"],
        json_path=config["logging"]["experiments_json"],
    )


if __name__ == "__main__":
    main()

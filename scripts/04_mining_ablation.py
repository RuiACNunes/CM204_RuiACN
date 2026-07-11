"""Layer 2 — ablação de estratégias de mineração de triplos.

Protocolo de variável única: dataset, split, head, loss, sampler e avaliação
fixos — varia apenas a estratégia de seleção de triplos. Produz quatro linhas
em experiments.csv e quatro curvas de convergência em results/.

Reutiliza `src.training.head_trainer.train_head` diretamente, sem refatoração.
O único pré-requisito é que o cache da Layer 0 exista
(`data/embeddings/clip_vitb16.npz`).

Uso:
    python scripts/04_mining_ablation.py --config configs/default.yaml
    python scripts/04_mining_ablation.py --strategies semi_hard distance_weighted
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.cache.embeddings import load_cached_embeddings
from src.training.head_trainer import train_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Sobrescreve a lista default ['batch_all','batch_hard','semi_hard','distance_weighted']",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    embeddings_path = os.path.join(
        config["paths"]["embeddings_dir"],
        config["backbone"]["embedding_filename"],
    )
    payload = load_cached_embeddings(embeddings_path)
    class_id = payload["class_id"].numpy()

    split_cfg = config["split"]
    train_lo, train_hi = split_cfg["train_classes"]
    test_lo, test_hi = split_cfg["test_classes"]
    train_mask = (class_id >= train_lo) & (class_id <= train_hi)
    test_mask = (class_id >= test_lo) & (class_id <= test_hi)

    # Todas as quatro estratégias — inclui as da Layer 1 para a tabela ser completa
    all_strategies = ["batch_all", "batch_hard", "semi_hard", "distance_weighted"]
    strategies = args.strategies or all_strategies

    for strategy in strategies:
        print(f"\n=== Layer 2 — mineração '{strategy}' ===")
        metrics = train_head(config, payload, train_mask, test_mask, strategy)
        print(f"    Resultado final [{strategy}]: {metrics}")


if __name__ == "__main__":
    main()

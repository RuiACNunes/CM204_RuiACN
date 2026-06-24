"""Layer 1 — treina a projection head sobre os embeddings congelados.

Driver fino: carrega os embeddings cacheados, monta as máscaras de split e
delega o treino propriamente dito a `src.training.head_trainer.train_head`,
chamada uma vez por estratégia de mineração configurada em
`training.mining_strategies` (default: batch_hard e batch_all — SPEC §6.5).

Uso:
    python scripts/02_train_head.py [--config configs/default.yaml]
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
        help="Sobrescreve training.mining_strategies (ex.: --strategies batch_hard)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    backbone_cfg = config["backbone"]
    embeddings_path = os.path.join(config["paths"]["embeddings_dir"], backbone_cfg["embedding_filename"])
    payload = load_cached_embeddings(embeddings_path)
    class_id = payload["class_id"].numpy()

    split_cfg = config["split"]
    train_lo, train_hi = split_cfg["train_classes"]
    test_lo, test_hi = split_cfg["test_classes"]
    train_mask = (class_id >= train_lo) & (class_id <= train_hi)
    test_mask = (class_id >= test_lo) & (class_id <= test_hi)

    strategies = args.strategies or config["training"]["mining_strategies"]
    for strategy in strategies:
        print(f"=== Treinando head com mineração '{strategy}' ===")
        train_head(config, payload, train_mask, test_mask, strategy)


if __name__ == "__main__":
    main()

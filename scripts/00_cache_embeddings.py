"""Layer 0 — extrai e cacheia os embeddings do CLIP congelado para todo o CUB.

Uso:
    python scripts/00_cache_embeddings.py [--config configs/default.yaml]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.cache.embeddings import cache_embeddings
from src.data.cub import load_cub
from src.models.backbone import ClipBackbone
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    backbone_cfg = config["backbone"]
    backbone = ClipBackbone(
        model_name=backbone_cfg["model_name"],
        pretrained=backbone_cfg["pretrained"],
        device=backbone_cfg["device"],
    )

    dataset = load_cub(config, transform=backbone.preprocess)
    print(f"Dataset CUB carregado: {len(dataset)} imagens.")

    output_path = os.path.join(
        config["paths"]["embeddings_dir"], backbone_cfg["embedding_filename"]
    )
    payload = cache_embeddings(
        dataset,
        backbone,
        output_path=output_path,
        batch_size=backbone_cfg["batch_size"],
        num_workers=backbone_cfg["num_workers"],
    )
    print(f"Embeddings salvos em {output_path}: {tuple(payload['embeddings'].shape)}")


if __name__ == "__main__":
    main()

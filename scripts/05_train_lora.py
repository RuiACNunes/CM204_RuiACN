"""Layer 3 — fine-tuning via LoRA do backbone CLIP.

Regime: lora_triplet_<strategy>

Diferença estrutural vs. Layers 0-2: o backbone ENTRA no grafo. O forward
passa por imagens (não por embeddings cacheados) — os embeddings mudam a
cada passo de otimização, tornando o cache inútil. Exige GPU.

O que é treinado:
  - Adaptadores LoRA injetados em out_proj de cada bloco de atenção do ViT
  - ProjectionHead (mesma arquitetura da Layer 1)
  O restante do backbone fica congelado (PEFT: <<1% dos parâmetros totais).

Avaliação: passa todo o split de teste pelo backbone adaptado + head a cada
eval_every épocas — sem cache, porque os embeddings evoluem.

Uso:
    python scripts/05_train_lora.py --config configs/default.yaml
    python scripts/05_train_lora.py --strategy batch_hard --rank 8
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.cub import load_cub, metric_learning_split
from src.data.sampler import PKSampler
from src.eval.metrics import evaluate_retrieval
from src.eval.retrieval import leave_one_out_ranking
from src.losses.triplet import TripletLoss
from src.mining.miners import mine_triplets
from src.models.backbone import ClipBackbone
from src.models.lora import LoRAConfig, count_parameters, inject_lora
from src.models.projection import ProjectionHead
from src.utils.logging import log_experiment
from src.utils.seed import set_seed


def _append_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _encode_images(loader: DataLoader, backbone: ClipBackbone, head: ProjectionHead, device: str) -> tuple:
    """Forward de inferência (sem gradiente) para avaliação periódica."""
    backbone.model.eval()
    head.eval()
    all_proj, all_labels = [], []
    with torch.no_grad():
        for imgs, class_ids, _ in loader:
            emb = backbone.model.encode_image(imgs.to(device, non_blocking=True)).float()
            proj = head(emb)
            all_proj.append(proj.cpu().numpy())
            all_labels.append(class_ids.numpy())
    return np.vstack(all_proj), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--strategy", default=None,
        help="Estratégia de mineração (default: lora.mining_strategy no config)"
    )
    parser.add_argument(
        "--rank", type=int, default=None,
        help="Rank LoRA (default: lora.rank no config) — para sweep de r"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    lora_cfg = config["lora"]
    train_cfg = config["training"]
    retrieval_cfg = config["retrieval"]
    backbone_cfg = config["backbone"]
    head_cfg = config["head"]
    sampler_cfg = config["sampler"]

    strategy = args.strategy or lora_cfg["mining_strategy"]
    rank = args.rank or lora_cfg["rank"]

    set_seed(config["seed"])

    # --- Backbone + LoRA -------------------------------------------------------
    backbone = ClipBackbone(
        model_name=backbone_cfg["model_name"],
        pretrained=backbone_cfg["pretrained"],
        device=backbone_cfg["device"],
    )
    device = backbone.device
    print(f"Device: {device}")

    lora_config = LoRAConfig(rank=rank, alpha=lora_cfg["alpha"], dropout=lora_cfg["dropout"])
    inject_lora(backbone.model, lora_config)
    backbone.model.to(device)  # defesa em profundidade: garante adaptadores no device

    trainable, total = count_parameters(backbone.model)
    print(
        f"Backbone — treináveis: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.3f}%)"
    )

    # --- Projection head -------------------------------------------------------
    head = ProjectionHead(
        in_dim=head_cfg["in_dim"],
        hidden_dim=head_cfg["hidden_dim"],
        out_dim=head_cfg["out_dim"],
    ).to(device)

    # --- Dataset ---------------------------------------------------------------
    dataset = load_cub(config, transform=backbone.preprocess)
    samples = dataset.samples
    train_idx, test_idx = metric_learning_split(
        samples,
        train_classes=tuple(config["split"]["train_classes"]),
        test_classes=tuple(config["split"]["test_classes"]),
    )

    train_labels = np.array([samples[i].class_id for i in train_idx])
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    P, K = sampler_cfg["P"], sampler_cfg["K"]
    num_workers = lora_cfg.get("num_workers", 2)

    pk_sampler = PKSampler(train_labels, P=P, K=K)
    train_loader = DataLoader(
        train_subset, sampler=pk_sampler, batch_size=P * K,
        num_workers=num_workers, pin_memory=(device != "cpu"),
    )
    test_loader = DataLoader(
        test_subset, batch_size=backbone_cfg["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=(device != "cpu"),
    )

    # --- Optimizer (dois param groups) -----------------------------------------
    # LoRA adapters: lr menor — partem de pesos pré-treinados
    # Projection head: lr maior — parte do zero
    lora_params = [p for p in backbone.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([
        {"params": lora_params,          "lr": lora_cfg["lr_backbone"]},
        {"params": head.parameters(),    "lr": lora_cfg["lr_head"]},
    ])

    triplet_loss = TripletLoss(margin=train_cfg["margin"])

    regime = f"lora_triplet_{strategy}_r{rank}"
    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    training_log_path = os.path.join(results_dir, f"training_{regime}.csv")

    # --- Loop de treino --------------------------------------------------------
    epochs = lora_cfg["epochs"]
    eval_every = lora_cfg["eval_every"]
    metrics: dict = {}

    for epoch in range(1, epochs + 1):
        # Backbone em eval (mantém BN/Dropout frozen); LoRA e head em train
        backbone.model.eval()
        head.train()

        epoch_loss, n_batches = 0.0, 0
        for imgs, batch_labels, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device)

            # Gradiente flui pelos adaptadores LoRA — NÃO usar backbone.encode()
            # (decorado com @torch.no_grad()). Chamada direta ao modelo:
            emb = backbone.model.encode_image(imgs).float()
            proj = head(emb)

            a_idx, p_idx, n_idx = mine_triplets(
                proj, batch_labels, strategy=strategy, margin=train_cfg["margin"]
            )
            loss = triplet_loss(proj, a_idx, p_idx, n_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % eval_every == 0 or epoch == epochs:
            test_proj, test_labels = _encode_images(test_loader, backbone, head, device)
            ranking = leave_one_out_ranking(test_proj, metric=retrieval_cfg["metric"])
            metrics = evaluate_retrieval(ranking, test_labels, retrieval_cfg["recall_ks"])
            _append_row(training_log_path, {"epoch": epoch, "loss": avg_loss, **metrics})
            # Volta a train mode após avaliação
            head.train()
            print(
                f"  [{regime}] epoch {epoch}/{epochs} — "
                f"loss: {avg_loss:.4f} — {metrics}"
            )

    # --- Log final -------------------------------------------------------------
    log_experiment(
        regime=regime,
        backbone=f"{backbone_cfg['model_name']}/{backbone_cfg['pretrained']}",
        loss="triplet",
        mining_strategy=strategy,
        hyperparams={
            "P": P,
            "K": K,
            "margin": train_cfg["margin"],
            "lr_backbone": lora_cfg["lr_backbone"],
            "lr_head": lora_cfg["lr_head"],
            "epochs": epochs,
            "dim": head_cfg["out_dim"],
            "lora_rank": rank,
            "lora_alpha": lora_cfg["alpha"],
        },
        l2_normalize=True,
        seed=config["seed"],
        metrics=metrics,
        csv_path=config["logging"]["experiments_csv"],
        json_path=config["logging"]["experiments_json"],
    )


if __name__ == "__main__":
    main()

"""[CM-204] Loop de treino reusável da projection head (SPEC §6).

Extraído de `scripts/02_train_head.py` para reuso direto pela Layer 2
(`scripts/04_mining_ablation.py`, SPEC §7): uma vez que `src/mining/miners.py`
implementar `semi_hard`/`distance_weighted`, a ablação de mineração só
precisa chamar `train_head(...)` variando `strategy` — mesmo formato já usado
pela Layer 1.

`train_head` reseta a seed global no início, de forma que cada chamada
(cada regime/estratégia) seja independentemente reprodutível e não herde o
estado do RNG de uma chamada anterior.
"""
import csv
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.sampler import PKSampler
from src.eval.metrics import evaluate_retrieval
from src.eval.retrieval import leave_one_out_ranking
from src.losses.triplet import TripletLoss
from src.mining.miners import mine_triplets
from src.models.projection import ProjectionHead
from src.utils.logging import log_experiment
from src.utils.seed import set_seed


def _append_training_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_head(
    config: dict,
    payload: dict,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    strategy: str,
    training_log_dir: Optional[str] = None,
) -> dict:
    """Treina uma `ProjectionHead` do zero com mineração `strategy` sobre os
    embeddings cacheados (raw) do split de treino, avaliando periodicamente no
    split de teste com o protocolo leave-one-out (SPEC §6.5).

    Grava:
      - uma linha por checkpoint de avaliação em
        `<training_log_dir>/training_head_triplet_<strategy>.csv` (curva de
        convergência: epoch, loss, R@1, R@2, R@4, R@8, mAP@R) — material para
        a análise ligada à teoria no relatório.
      - uma linha final (métricas do último checkpoint) em
        `results/experiments.csv` / `.json` via `log_experiment`.

    Retorna o dict de métricas do último checkpoint avaliado.
    """
    set_seed(config["seed"])

    device = config["training"]["device"]
    train_mask_t = torch.from_numpy(train_mask)
    test_mask_t = torch.from_numpy(test_mask)

    train_embeddings = payload["embeddings"][train_mask_t].to(device)
    train_labels = payload["class_id"][train_mask_t].to(device)
    test_embeddings = payload["embeddings"][test_mask_t].numpy()
    test_labels = payload["class_id"][test_mask_t].numpy()

    head_cfg = config["head"]
    head = ProjectionHead(
        in_dim=head_cfg["in_dim"], hidden_dim=head_cfg["hidden_dim"], out_dim=head_cfg["out_dim"]
    ).to(device)

    sampler_cfg = config["sampler"]
    P, K = sampler_cfg["P"], sampler_cfg["K"]
    sampler = PKSampler(train_labels.cpu().numpy(), P=P, K=K)
    dataset = TensorDataset(train_embeddings.cpu(), train_labels.cpu())
    loader = DataLoader(dataset, sampler=sampler, batch_size=P * K)

    train_cfg = config["training"]
    optimizer = torch.optim.Adam(head.parameters(), lr=train_cfg["lr"])
    triplet_loss = TripletLoss(margin=train_cfg["margin"])
    retrieval_cfg = config["retrieval"]

    regime = f"head_triplet_{strategy}"
    log_dir = training_log_dir or config["paths"]["results_dir"]
    training_log_path = os.path.join(log_dir, f"training_{regime}.csv")

    def _evaluate(epoch: int, avg_loss: float) -> dict:
        head.eval()
        with torch.no_grad():
            test_proj = (
                head(torch.tensor(test_embeddings, dtype=torch.float32, device=device)).cpu().numpy()
            )
        ranking = leave_one_out_ranking(test_proj, metric=retrieval_cfg["metric"])
        eval_metrics = evaluate_retrieval(ranking, test_labels, retrieval_cfg["recall_ks"])
        _append_training_row(training_log_path, {"epoch": epoch, "loss": avg_loss, **eval_metrics})
        head.train()
        return eval_metrics

    metrics: dict = {}
    for epoch in range(1, train_cfg["epochs"] + 1):
        head.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_emb, batch_labels in loader:
            batch_emb = batch_emb.to(device)
            batch_labels = batch_labels.to(device)

            proj = head(batch_emb)
            a_idx, p_idx, n_idx = mine_triplets(proj, batch_labels, strategy=strategy)
            loss = triplet_loss(proj, a_idx, p_idx, n_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % train_cfg["eval_every"] == 0 or epoch == train_cfg["epochs"]:
            metrics = _evaluate(epoch, avg_loss)
            print(f"  [{strategy}] epoch {epoch}/{train_cfg['epochs']} — loss média: {avg_loss:.4f} — {metrics}")

    log_experiment(
        regime=regime,
        backbone=f"{config['backbone']['model_name']}/{config['backbone']['pretrained']}",
        loss="triplet",
        mining_strategy=strategy,
        hyperparams={
            "P": P,
            "K": K,
            "margin": train_cfg["margin"],
            "lr": train_cfg["lr"],
            "epochs": train_cfg["epochs"],
            "dim": head_cfg["out_dim"],
        },
        l2_normalize=True,  # saída da head é estruturalmente L2-normalizada (SPEC §2.1)
        seed=config["seed"],
        metrics=metrics,
        csv_path=config["logging"]["experiments_csv"],
        json_path=config["logging"]["experiments_json"],
    )
    return metrics

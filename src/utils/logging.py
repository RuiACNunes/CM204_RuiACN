"""[VIAL-infra] Log estruturado de experimentos em CSV/JSON.

Cada chamada a `log_experiment` grava uma linha em `results/experiments.csv`
(schema fixo) e acrescenta o mesmo registro a `results/experiments.json`
(lista de objetos). A tabela do relatório IEEE (SPEC §6.6) é montada
diretamente a partir do CSV, sem recompor números à mão (SPEC §9).
"""
import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Schema fixo: mantém o CSV com colunas estáveis mesmo quando um regime não usa
# todos os hiperparâmetros (campos não aplicáveis ficam vazios).
FIELDNAMES = [
    "timestamp",
    "regime",
    "backbone",
    "loss",
    "mining_strategy",
    "P",
    "K",
    "margin",
    "lr",
    "epochs",
    "dim",
    "l2_normalize",
    "seed",
    "R@1",
    "R@2",
    "R@4",
    "R@8",
    "mAP@R",
]


def log_experiment(
    regime: str,
    backbone: str,
    loss: Optional[str],
    mining_strategy: Optional[str],
    hyperparams: Dict[str, Any],
    l2_normalize: bool,
    seed: int,
    metrics: Dict[str, float],
    csv_path: str,
    json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Grava uma linha de resultado de experimento.

    `hyperparams` pode conter qualquer subconjunto de
    {P, K, margin, lr, epochs, dim} — chaves ausentes ficam vazias no CSV.
    `metrics` deve conter as chaves R@1, R@2, R@4, R@8, mAP@R (algumas podem
    faltar se não computadas).
    """
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "backbone": backbone,
        "loss": loss,
        "mining_strategy": mining_strategy,
        "P": hyperparams.get("P"),
        "K": hyperparams.get("K"),
        "margin": hyperparams.get("margin"),
        "lr": hyperparams.get("lr"),
        "epochs": hyperparams.get("epochs"),
        "dim": hyperparams.get("dim"),
        "l2_normalize": l2_normalize,
        "seed": seed,
        "R@1": metrics.get("R@1"),
        "R@2": metrics.get("R@2"),
        "R@4": metrics.get("R@4"),
        "R@8": metrics.get("R@8"),
        "mAP@R": metrics.get("mAP@R"),
    }

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    if json_path is not None:
        records = []
        if os.path.isfile(json_path):
            with open(json_path, "r") as f:
                try:
                    records = json.load(f)
                except json.JSONDecodeError:
                    records = []
        records.append(row)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2, default=str)

    return row

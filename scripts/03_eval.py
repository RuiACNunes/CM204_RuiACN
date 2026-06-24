"""Avaliação unificada — consolida `results/experiments.csv` na tabela final
de métricas (mesmas colunas para todos os regimes, SPEC §6.6).

Não recomputa nada: apenas lê o log estruturado preenchido pelos scripts 01 e
02 e formata a tabela usada diretamente no relatório IEEE.

Uso:
    python scripts/03_eval.py [--config configs/default.yaml]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml

COLUMNS = ["regime", "R@1", "R@2", "R@4", "R@8", "mAP@R"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default=None, help="Caminho do .md de saída (default: results/summary_table.md)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    csv_path = config["logging"]["experiments_csv"]
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"{csv_path} não encontrado. Rode scripts/01_baseline_knn.py e scripts/02_train_head.py antes."
        )

    df = pd.read_csv(csv_path)
    # mantém a última execução de cada regime (reexecuções sobrescrevem a anterior na tabela final)
    df = df.drop_duplicates(subset="regime", keep="last")
    table = df[COLUMNS].rename(columns={"regime": "Regime"})

    print(table.to_string(index=False))

    output_path = args.output or os.path.join(config["paths"]["results_dir"], "summary_table.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table.to_markdown(index=False))
    print(f"\nTabela salva em {output_path}")


if __name__ == "__main__":
    main()

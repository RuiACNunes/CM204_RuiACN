"""Layer 2 — estudo de mineração (estrutura agora, implementação futura).

Planejado (SPEC §7): roda os mesmos regimes da Layer 1 variando apenas a
estratégia de mineração em `src/mining/miners.py` (`semi_hard`,
`distance_weighted`, além de `batch_hard`/`batch_all` já implementadas),
produzindo uma tabela comparativa — eixo experimental "de graça" sobre a
Layer 1.

Quando essas estratégias forem implementadas, este script só precisa chamar
`src.training.head_trainer.train_head(config, payload, train_mask, test_mask,
strategy)` para cada uma — mesmo formato já usado por `scripts/02_train_head.py`
para `batch_hard`/`batch_all`; nenhuma refatoração de treino é necessária.

Fora do escopo do protótipo pré-26/06 (SPEC §11): este script é apenas o
esqueleto, não é executado.
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.parse_args()
    raise NotImplementedError(
        "Layer 2 (ablação de mineração) ainda não implementada — ver SPEC §7. "
        "Requer antes estender src/mining/miners.py com 'semi_hard' e 'distance_weighted'; "
        "depois, basta chamar src.training.head_trainer.train_head(...) variando strategy."
    )


if __name__ == "__main__":
    main()

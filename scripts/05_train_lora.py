"""Layer 3 — adaptação via LoRA (estrutura agora, implementação futura).

Planejado (SPEC §8): mesmo protocolo de avaliação das Layers 0-1, com um novo
regime `lora_triplet_batch_hard`. O backbone passa a entrar no grafo (forward
sobre imagens, não mais sobre embeddings cacheados) — exige GPU de verdade
(Kaggle como ambiente principal). É o resultado central planejado do projeto;
entra após aprovação do tema.

Fora do escopo do protótipo pré-26/06 (SPEC §11): este script é apenas o
esqueleto, não é executado.
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.parse_args()
    raise NotImplementedError(
        "Layer 3 (LoRA) ainda não implementada — ver SPEC §8. "
        "Requer antes implementar src/models/lora.py:inject_lora()."
    )


if __name__ == "__main__":
    main()

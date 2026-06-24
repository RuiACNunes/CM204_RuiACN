"""[CM-204] PK sampler para metric learning.

Cada batch contém P classes x K imagens por classe (default P=8, K=4 -> batch
32), garantindo positivos e negativos em todo batch para a mineração de
triplos (SPEC §6.2). Implementação própria de `torch.utils.data.Sampler`.

Uso: `DataLoader(dataset, sampler=PKSampler(labels, P, K), batch_size=P*K)`.
O sampler devolve uma lista "achatada" de índices; cada bloco consecutivo de
P*K índices forma um batch PK válido.
"""
from typing import List, Sequence

import numpy as np
from torch.utils.data import Sampler


class PKSampler(Sampler[int]):
    def __init__(self, labels: Sequence[int], P: int, K: int):
        self.labels = np.asarray(labels)
        self.P = P
        self.K = K
        self.classes = np.unique(self.labels)
        if len(self.classes) < P:
            raise ValueError(
                f"PKSampler: P={P} excede o número de classes disponíveis ({len(self.classes)})."
            )
        self.class_to_indices = {c: np.where(self.labels == c)[0] for c in self.classes}
        self.num_batches = len(self.labels) // (P * K)

    def __len__(self) -> int:
        return self.num_batches * self.P * self.K

    def __iter__(self):
        flat: List[int] = []
        for _ in range(self.num_batches):
            chosen_classes = np.random.choice(self.classes, size=self.P, replace=False)
            for c in chosen_classes:
                idxs = self.class_to_indices[c]
                replace = len(idxs) < self.K
                chosen = np.random.choice(idxs, size=self.K, replace=replace)
                flat.extend(chosen.tolist())
        return iter(flat)

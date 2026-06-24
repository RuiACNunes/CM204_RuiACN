"""[VIAL-infra: motivação temática] Carga do CUB-200-2011 e split de metric learning.

Fonte default e única plenamente suportada: **"local"**, contra a estrutura
oficial do `CUB_200_2011.tgz` do Caltech — reproduzível e determinística, sem
depender de um mirror de terceiros. Usa apenas três arquivos do pacote
oficial:
  - `images.txt`             (image_id -> caminho relativo)
  - `image_class_labels.txt` (image_id -> class_id, 1-200)
  - `classes.txt`            (class_id -> nome da classe)

`train_test_split.txt` é **deliberadamente ignorado**: aquele é o split de
classificação original do CUB, e o split deste estudo é por classe (1-100
treino, 101-200 teste; ver `metric_learning_split`), não por imagem.

Fonte "hf" (Hugging Face Hub via `datasets`) existe apenas como **fallback
opcional, não-default**: o nome exato do mirror e o schema de campos não
foram confirmados (ver `data.hf_dataset_name` em configs/default.yaml).
Verifique-os antes de usar `source: "hf"`.
"""
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class CUBSample:
    class_id: int
    image_id: int
    image_path: Optional[str] = None  # fonte "local"
    hf_index: Optional[int] = None  # fonte "hf"


def _load_local_index(root: str) -> Tuple[List[CUBSample], Dict[int, str]]:
    """Lê o índice de imagens da estrutura oficial do CUB_200_2011.tgz extraído.

    Espera em `root`: images.txt, image_class_labels.txt, classes.txt, e a
    pasta images/. `train_test_split.txt` não é lido (ver docstring do módulo).
    """
    images_txt = os.path.join(root, "images.txt")
    labels_txt = os.path.join(root, "image_class_labels.txt")
    classes_txt = os.path.join(root, "classes.txt")
    images_dir = os.path.join(root, "images")

    for path in (images_txt, labels_txt, classes_txt):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Arquivo esperado do CUB-200-2011 não encontrado: {path}. "
                f"Extraia CUB_200_2011.tgz em '{root}' (ver README, seção de execução)."
            )

    image_paths = {}
    with open(images_txt, "r") as f:
        for line in f:
            image_id, rel_path = line.strip().split(" ", 1)
            image_paths[int(image_id)] = os.path.join(images_dir, rel_path)

    class_ids = {}
    with open(labels_txt, "r") as f:
        for line in f:
            image_id, class_id = line.strip().split()
            class_ids[int(image_id)] = int(class_id)

    class_names: Dict[int, str] = {}
    with open(classes_txt, "r") as f:
        for line in f:
            class_id, class_name = line.strip().split(" ", 1)
            class_names[int(class_id)] = class_name

    samples = []
    for image_id in sorted(image_paths):
        samples.append(
            CUBSample(
                class_id=class_ids[image_id],
                image_id=image_id,
                image_path=image_paths[image_id],
            )
        )
    return samples, class_names


def _load_hf_index(dataset_name: str, label_field: str, zero_indexed: bool):
    """Carrega um mirror do CUB-200-2011 no Hugging Face Hub (fallback opcional).

    Retorna (hf_dataset, samples). Ver ATENÇÃO no docstring do módulo: o nome e
    o schema do dataset não foram confirmados — confira antes de usar.
    """
    from datasets import concatenate_datasets, load_dataset

    ds = load_dataset(dataset_name)
    if hasattr(ds, "keys"):
        ds = concatenate_datasets([ds[split] for split in ds.keys()])

    samples = []
    for i, label in enumerate(ds[label_field]):
        class_id = int(label) + 1 if zero_indexed else int(label)
        samples.append(CUBSample(class_id=class_id, image_id=i, hf_index=i))
    return ds, samples


class CUBDataset(Dataset):
    """Dataset CUB-200-2011 — devolve (image_tensor, class_id, image_id)."""

    def __init__(
        self,
        samples: List[CUBSample],
        transform: Optional[Callable] = None,
        hf_dataset=None,
        hf_image_field: str = "image",
        class_names: Optional[Dict[int, str]] = None,
    ):
        self.samples = samples
        self.transform = transform
        self.hf_dataset = hf_dataset
        self.hf_image_field = hf_image_field
        self.class_names = class_names or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        if sample.image_path is not None:
            image = Image.open(sample.image_path).convert("RGB")
        else:
            image = self.hf_dataset[sample.hf_index][self.hf_image_field].convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.class_id, sample.image_id


def load_cub(config: dict, transform: Optional[Callable] = None) -> CUBDataset:
    """Constrói o `CUBDataset` completo (11.788 imagens) a partir da config."""
    data_cfg = config["data"]
    if data_cfg["source"] == "local":
        samples, class_names = _load_local_index(config["paths"]["cub_root"])
        return CUBDataset(samples, transform=transform, class_names=class_names)
    elif data_cfg["source"] == "hf":
        hf_dataset, samples = _load_hf_index(
            data_cfg["hf_dataset_name"],
            data_cfg["hf_label_field"],
            data_cfg["hf_label_is_zero_indexed"],
        )
        return CUBDataset(
            samples,
            transform=transform,
            hf_dataset=hf_dataset,
            hf_image_field=data_cfg["hf_image_field"],
        )
    else:
        raise ValueError(f"data.source desconhecido: {data_cfg['source']!r} (use 'local' ou 'hf')")


def metric_learning_split(
    samples: List[CUBSample],
    train_classes: Tuple[int, int] = (1, 100),
    test_classes: Tuple[int, int] = (101, 200),
) -> Tuple[List[int], List[int]]:
    """Split do estudo (SPEC §2/§5.1): classes `train_classes` -> treino,
    classes `test_classes` -> teste. Disjunto e determinístico (sem amostragem
    aleatória). Retorna (train_indices, test_indices) sobre `samples`.
    """
    train_idx, test_idx = [], []
    for i, s in enumerate(samples):
        if train_classes[0] <= s.class_id <= train_classes[1]:
            train_idx.append(i)
        elif test_classes[0] <= s.class_id <= test_classes[1]:
            test_idx.append(i)
    return train_idx, test_idx

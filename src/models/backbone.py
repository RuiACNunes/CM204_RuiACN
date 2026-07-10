"""[VIAL-infra: motivação temática] Backbone CLIP congelado via `open_clip`.

Único backbone do projeto (SPEC §2): CLIP, default ViT-B/16 com pesos OpenAI,
totalmente congelado. `encode()` devolve o embedding global nativo do
`encode_image` **sem normalizar** (1 vetor por imagem, sem mean-pooling de
patches) — `open_clip.encode_image` já devolve features não normalizadas por
padrão, e cachear raw é o caminho natural.

A normalização é decisão de cada consumidor do embedding (baseline kNN vs.
input da projection head), não do backbone — ver `src/cache/embeddings.py` e
`scripts/01_baseline_knn.py`.
"""
import warnings

import open_clip
import torch


def _resolve_device(device: str) -> str:
    """Resolve "auto" para o melhor dispositivo disponível.

    Permite que o mesmo `configs/default.yaml` rode em CPU no laptop
    (Layers 0-1, sobre cache) e em CUDA no Colab/Kaggle (extração, LoRA),
    sem edição manual.
    """
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


def _resolve_model_name(model_name: str, pretrained: str) -> str:
    """Os pesos originais da OpenAI foram treinados com ativação QuickGELU.

    Carregá-los sob um nome que não sinaliza isso (`ViT-B-16` em vez de
    `ViT-B-16-quickgelu`) faz o `open_clip` emitir um aviso de mismatch e
    instanciar a ativação errada — os embeddings degradam **silenciosamente**,
    sem exceção. Corrigimos o nome antes de instanciar.
    """
    if pretrained == "openai" and not model_name.endswith("-quickgelu"):
        resolved = f"{model_name}-quickgelu"
        warnings.warn(
            f"pretrained='openai' exige ativação QuickGELU; "
            f"usando '{resolved}' em vez de '{model_name}'.",
            stacklevel=2,
        )
        return resolved
    return model_name


class ClipBackbone:
    """[VIAL-infra] Backbone CLIP congelado. Não treina; apenas extrai embeddings.

    O embedding devolvido é a **saída global nativa** de `encode_image` (um vetor
    por imagem, derivado do CLS e projetado), **sem** mean-pooling de patches e
    **sem** normalização L2 — ver `encode`.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        device: str = "auto",
    ):
        self.model_name = _resolve_model_name(model_name, pretrained)
        self.pretrained = pretrained
        self.device = _resolve_device(device)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=pretrained
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.to(self.device)
        self.embed_dim = self.model.visual.output_dim

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: Tensor[N, C, H, W] já pré-processado por `self.preprocess`.

        Retorna Tensor[N, D] em float32, no device do backbone — embedding global
        nativo do CLIP, **raw** (não normalizado). A normalização L2 é aplicada no
        ponto de consumo: no baseline kNN (conforme `retrieval.baseline_l2_normalize`)
        ou na saída da projection head (onde é estrutural). Manter o cache raw
        desacopla essas decisões e evita recachear ao variar a ablação.
        """
        images = images.to(self.device, non_blocking=True)
        return self.model.encode_image(images).float()

    def __repr__(self) -> str:
        return (
            f"ClipBackbone(model_name={self.model_name!r}, "
            f"pretrained={self.pretrained!r}, device={self.device!r}, "
            f"embed_dim={self.embed_dim})"
        )
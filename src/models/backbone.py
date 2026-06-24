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
import open_clip
import torch


class ClipBackbone:
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        device: str = "cpu",
    ):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.device = device
        self.model.to(device)
        self.embed_dim = self.model.visual.output_dim

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: Tensor[N, C, H, W] já pré-processado por `self.preprocess`.

        Retorna Tensor[N, D] — embedding global nativo do CLIP, **raw** (não
        normalizado). Normalize no ponto de consumo, se necessário.
        """
        images = images.to(self.device)
        return self.model.encode_image(images)

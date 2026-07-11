# Implementação — Layers 2 e 3

## Layer 2 · Ablação de mineração (`scripts/04_mining_ablation.py`)

Roda as quatro estratégias de mineração sobre o **cache da Layer 0** (`data/embeddings/clip_vitb16.npz`). O backbone não entra no grafo — tudo roda em CPU sobre embeddings fixos, como na Layer 1.

### Estratégias implementadas em `src/mining/miners.py`

**`batch_hard` / `batch_all`** — já existiam na Layer 1, incluídas na ablação para a tabela ficar completa.

**`semi_hard`** (FaceNet, Schroff 2015):
```python
# Para cada par (âncora i, positivo p):
d_ap = dist[i, p]
candidatos = negativos com d_ap < d(i,n) < d_ap + margin
# Se não houver candidato → fallback para o negativo mais difícil (batch_hard)
# Se houver → sorteia aleatoriamente entre os candidatos
```
Política de fallback escolhida: hardest negative (opção mais robusta nas primeiras épocas).

**`distance_weighted`** (Wu 2017):
```python
# Para cada âncora i, computa log-peso dos negativos:
log_w = -(D-2)*log(d) - (D-3)/2 * log(1 - d²/4)
# D = dimensão do embedding (128 para a head)
# Clamp: d_min=0.5 (evita explosão), nonzero_loss_cutoff=1.4 (descarta triviais)
# Amostra um negativo via multinomial(softmax(log_w))
# Fallback para batch_hard se todos além do cutoff
```

### Interface `mine_triplets`
```python
mine_triplets(embeddings, labels, strategy="batch_hard", margin=0.2)
# margin só é usado por semi_hard
# backward-compatible: batch_hard/batch_all ignoram margin
```

### Como rodar
```bash
# Todas as 4 estratégias (sequencial, ~4× o tempo da Layer 1)
python scripts/04_mining_ablation.py --config configs/default.yaml

# Só as novas:
python scripts/04_mining_ablation.py --strategies semi_hard distance_weighted
```

**Saída:** quatro linhas em `results/experiments.csv` + quatro CSVs de convergência `results/training_head_triplet_<strategy>.csv`.

---

## Layer 3 · LoRA (`scripts/05_train_lora.py`)

### Diferença estrutural vs. Layers 0-2

| | Layers 0–2 | Layer 3 |
|---|---|---|
| Entrada do treino | embeddings cacheados | **imagens** |
| Backbone no grafo | não | **sim** |
| GPU obrigatória | não | **sim** |
| Cache reutilizável | sim | não (embeddings mudam) |

### Injeção dos adaptadores — `src/models/lora.py`

**`LoRALinear`**: substitui um `nn.Linear` por `W·x + (α/r)·B·A·x`
- `A` inicializado com Kaiming, `B` inicializado com **zeros** → delta=0 no passo 0, modelo parte do CLIP original.

**`inject_lora`**:
```python
# 1. Congela todo o backbone visual
for p in clip_model.visual.parameters():
    p.requires_grad_(False)

# 2. Substitui out_proj em cada bloco de atenção
for block in clip_model.visual.transformer.resblocks:
    block.attn.out_proj = LoRALinear(block.attn.out_proj, rank, alpha, dropout)
```

**Por que `out_proj` e não Q/V?** O open_clip funde Q, K, V num único tensor `in_proj_weight` (não é `nn.Linear`), impossibilitando substituição direta. `out_proj` é um `nn.Linear` exposto e pode ser substituído sem reimplementar o forward do `MultiheadAttention`. O paper de LoRA (Tabela 6) reporta resultados comparáveis entre out_proj e Q+V em ViTs.

**Escala de parâmetros** (ViT-B/16, rank=4):
- Treináveis: 12 blocos × 2 adaptadores × 2 × (768 × 4) ≈ **74k params**
- Total backbone: ~86M
- Fração: **~0.09%** — PEFT clássico

### Loop de treino

```python
# Dois param groups — escalas de gradiente diferentes:
optimizer = Adam([
    {"params": lora_params,       "lr": 1e-4},   # LoRA
    {"params": head.parameters(), "lr": 1e-3},   # head
])

for imgs, labels, _ in train_loader:
    # NÃO usar backbone.encode() — decorado com @torch.no_grad()
    emb = backbone.model.encode_image(imgs.to(device)).float()
    proj = head(emb)                              # [B, 128] L2-norm
    a, p, n = mine_triplets(proj, labels, strategy, margin)
    loss = TripletLoss()(proj, a, p, n)
    loss.backward()   # gradiente flui pelos adaptadores LoRA
    optimizer.step()
```

A `backbone.model.eval()` é mantida durante todo o treino (congela BN/Dropout dos layers base); os adaptadores LoRA e a head não têm BN, então o modo eval não os afeta.

### Avaliação periódica

A cada `eval_every` épocas, passa todo o split de teste pelo backbone adaptado + head (sem cache):
```python
with torch.no_grad():
    emb = backbone.model.encode_image(imgs).float()
    proj = head(emb)
# → leave_one_out_ranking → R@{1,2,4,8}, mAP@R
```

### Como rodar no Colab

```python
# Sweep de rank (ablação PEFT):
!python scripts/05_train_lora.py --config configs/default.yaml --rank 4
!python scripts/05_train_lora.py --rank 8
!python scripts/05_train_lora.py --rank 16

# Estratégia diferente (se Layer 2 apontar vencedora):
!python scripts/05_train_lora.py --strategy semi_hard
```

**Config relevante** (`configs/default.yaml`, seção `lora`):
```yaml
lora:
  rank: 4
  alpha: 8.0          # alpha = 2*rank (convenção padrão)
  dropout: 0.1
  lr_backbone: 1.0e-4
  lr_head: 1.0e-3
  epochs: 50
  eval_every: 5
  mining_strategy: "batch_hard"
  num_workers: 2      # Colab/Kaggle; 0 no Windows
```

**Saída:** linha `lora_triplet_<strategy>_r<rank>` em `results/experiments.csv` + curva `results/training_lora_triplet_<strategy>_r<rank>.csv`.

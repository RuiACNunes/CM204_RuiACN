# SPEC — CM-204 · Metric Learning com CLIP congelado

> **Propósito deste documento.** Blueprint de implementação. Será consumido pelo
> Claude Code dentro de uma IDE para gerar o código do projeto. Aqui ficam as
> decisões fechadas, a estrutura de arquivos e o contrato de cada módulo
> (entradas, saídas, hiperparâmetros). O Claude Code escreve a implementação a
> partir daqui — este arquivo é a fonte única de verdade das decisões.
>
> **Idioma:** prosa em português; identificadores de código em inglês.
> **Formato do projeto:** solo. **Deliverable final:** relatório IEEE de 8 páginas.

---

## 1. Identidade e pergunta de pesquisa

Estudo experimental de **metric learning** sobre um backbone **CLIP congelado**,
medindo o ganho de adicionar treino (cabeça de projeção e, no futuro, LoRA) sobre
a representação congelada, na tarefa de **recuperação de imagens em nível de
instância** no dataset **CUB-200-2011**.

O eixo experimental é um protocolo de **variável única**: a representação muda de
regime (congelada → head treinada → backbone adaptado via LoRA) enquanto dataset,
split, métricas e protocolo de avaliação permanecem fixos. Cada regime produz a
mesma tabela de métricas, tornando o ganho de cada incremento diretamente legível.

> **Nota de escopo (CLIP apenas).** No TG existe uma comparação CLIP vs. DINOv2.
> **Neste projeto usamos somente CLIP.** Nenhum código de DINOv2 entra aqui.

---

## 2. Decisões fechadas (fonte única de verdade)

| # | Decisão | Valor fechado |
|---|---------|---------------|
| Backbone | Único, congelado | **CLIP via `open_clip`**, default **ViT-B/16, pesos OpenAI** (configurável) |
| Embedding | Representação por imagem | **Saída global nativa** do `encode_image` (vetor único derivado do CLS, projetado). **Sem mean-pooling** de patches. Dim. típica 512 para ViT-B |
| Normalização | Espaço de busca/treino | `l2_normalize` é flag de config (ver §2.1) |
| Split | Protocolo de avaliação | **Metric learning:** primeiras 100 classes → treino; últimas 100 (disjuntas) → teste |
| Métricas | Reportadas | **Recall@{1,2,4,8}** e **mAP@R** (todas vão ao relatório) |
| Perda | Cabeça de projeção | **Triplet loss** (margem 0.2) |
| Mineração | Seleção de triplos | **Batch-hard** (default) + **batch-all** como ablação interna gratuita |
| Head | Arquitetura | **MLP 2 camadas**, BatchNorm + ReLU, saída **128-d**, L2-normalizada |
| Sampler | Amostragem de batch | **PK sampler** (P classes × K imagens), default P=8, K=4 |
| Logging | Resultados | Estruturado em **CSV/JSON** desde o 1º experimento; seed fixa |
| Docs | Desde o início | **README + requirements.txt**, atualizáveis incrementalmente |

### 2.1 Sobre a normalização (decisão com nuance)

Você indicou não ver necessidade de normalizar, mas se abriu a mudar. O spec
trata assim, por ser o mais defensável:

- **Saída da head:** **sempre L2-normalizada** (padrão em metric learning; a
  triplet loss com distância euclidiana sobre vetores normalizados equivale a
  distância de cosseno e é o regime estável). Isso é estrutural, não opcional.
- **Baseline congelado (Layer 0):** flag `baseline_l2_normalize`, **default
  `True`** (busca por cosseno, padrão da literatura de retrieval). Se quiser medir
  o efeito de não-normalizar, basta rodar com `False` — vira um ponto de ablação.

> Se você preferir mesmo começar sem normalizar o baseline, troque o default para
> `False`; o código já suporta os dois por flag.

---

## 3. Reúso e disclosure (fronteira VIAL / CM-204)

Mapeada diretamente na estrutura de arquivos, para o relatório citar sem zona cinzenta:

- **Herdado do VIAL (contexto temático + infraestrutura de recuperação):** a
  pergunta de pesquisa e a camada de *retrieval* (índice FAISS, busca kNN, métricas
  de recuperação). Concentrado em `src/eval/`.
- **Exclusivo de CM-204 (contribuição própria — onde mora a nota de código):** todo
  o **pipeline de treino** — cabeça de projeção, triplet loss, mineração,
  PK sampler e (futuro) LoRA. Concentrado em `src/models/projection.py`,
  `src/losses/`, `src/mining/`, `src/models/lora.py`.

> Frase-base para o relatório: *"O pipeline de treinamento (projection head, triplet
> loss, mineração de negativos e adaptação via LoRA) foi implementado exclusivamente
> para CM-204. Do contexto do VIAL herdam-se apenas a motivação temática e a
> infraestrutura de embeddings/recuperação."*

---

## 4. Estrutura de arquivos

```
cm204-metric-learning/
├── README.md                     # identidade, decisões, como rodar (ver §10)
├── requirements.txt              # dependências (maleável)
├── .gitignore                    # ignora data/ e results/*.csv pesados
├── configs/
│   └── default.yaml              # hiperparâmetros, paths, seed — fonte central
├── data/
│   └── .gitkeep                  # CUB + embeddings cacheados (gitignored)
├── results/
│   └── .gitkeep                  # logs de experimentos (CSV/JSON)
├── src/
│   ├── data/
│   │   ├── cub.py                # download/carga do CUB + split metric-learning
│   │   └── sampler.py            # PK sampler
│   ├── models/
│   │   ├── backbone.py           # CLIP congelado (open_clip) + encode_image
│   │   ├── projection.py         # [CM-204] MLP projection head
│   │   └── lora.py               # [CM-204] LoRA — Layer 3 (stub por enquanto)
│   ├── losses/
│   │   ├── triplet.py            # [CM-204] triplet loss (batch-hard / batch-all)
│   │   └── contrastive.py        # [CM-204] SupCon — ablação futura (stub)
│   ├── mining/
│   │   └── miners.py             # [CM-204] estratégias de mineração (param. `strategy`)
│   ├── eval/
│   │   ├── retrieval.py          # [VIAL-infra] índice FAISS + busca kNN (leave-one-out)
│   │   └── metrics.py            # [VIAL-infra] Recall@k, mAP@R
│   ├── cache/
│   │   └── embeddings.py         # extração e cache dos embeddings congelados
│   └── utils/
│       ├── seed.py               # reprodutibilidade (seed global)
│       └── logging.py            # log estruturado de experimentos → CSV/JSON
└── scripts/
    ├── 00_cache_embeddings.py    # Layer 0: extrai e cacheia embeddings do CLIP
    ├── 01_baseline_knn.py        # Layer 0: baseline congelado + kNN
    ├── 02_train_head.py          # Layer 1: treina a projection head
    ├── 03_eval.py                # avaliação unificada → tabela de métricas
    ├── 04_mining_ablation.py     # Layer 2 (futuro)
    └── 05_train_lora.py          # Layer 3 (futuro)
```

Tags `[CM-204]` e `[VIAL-infra]` nos comentários de cabeçalho de cada arquivo
reforçam a fronteira de disclosure do §3.

---

## 5. LAYER 0 — Baseline congelado + kNN  *(detalhar bem)*

Objetivo: estabelecer o piso de referência sem treino algum. É o número contra o
qual todo ganho será medido, e valida que dados, backbone e avaliação funcionam.

### 5.1 `src/data/cub.py`
- Carregar o CUB-200-2011 (preferir mirror Hugging Face `datasets`; suportar também
  pasta local extraída do `.tgz`). Não exige fotos do usuário — é download.
- Expor, por imagem: `image`, `class_id` (1–200), `image_id`, `is_train_split`
  (do split de classificação original — **não** usar este como split do estudo).
- **Split do estudo (metric learning):** função `metric_learning_split()` que
  devolve índices de treino = classes 1–100, teste = classes 101–200. Disjunto.
  Determinístico.

### 5.2 `src/models/backbone.py`
- Carregar CLIP via `open_clip.create_model_and_transforms(model_name, pretrained)`.
  Defaults: `model_name="ViT-B-16"`, `pretrained="openai"`.
- Congelar todos os parâmetros (`requires_grad=False`, `model.eval()`).
- `encode(images) -> Tensor[N, D]`: usa `model.encode_image`, devolve o **embedding
  global nativo** (1 vetor por imagem). Sem mean-pooling de patches. Aplica
  `l2_normalize` conforme flag.

### 5.3 `src/cache/embeddings.py`  ← chamado por `scripts/00_cache_embeddings.py`
- Passar **todas** as 11.788 imagens pelo backbone **uma única vez**.
- Salvar em `data/embeddings/clip_vitb16.{npy|pt}`: tensor `[N, D]` + arrays
  paralelos `class_id`, `image_id`, e máscara de split.
- A partir daqui, **nada mais precisa de GPU** nas Layers 0 e 1 (toda iteração roda
  sobre os embeddings cacheados, inclusive em CPU). Esta é a única etapa que vale
  acelerar com GPU (Colab/Kaggle) se a extração em CPU ficar lenta.

### 5.4 `src/eval/retrieval.py` + `src/eval/metrics.py`  *(infra herdada do VIAL)*
- **Protocolo:** avaliação **leave-one-out** sobre o split de teste. Cada embedding
  de teste é uma *query*; a *gallery* é o restante do split de teste (exclui a
  própria query). Não há gallery separada.
- `retrieval.py`: construir índice FAISS sobre os embeddings de teste e buscar os
  k vizinhos (exato; ~5.8k vetores é trivial — FAISS por consistência com a infra
  do VIAL, mas busca exata em torch seria equivalente nesta escala).
- `metrics.py`:
  - **Recall@k** (k ∈ {1,2,4,8}): fração de queries cujo top-k (excluindo a própria)
    contém ≥1 vizinho da mesma `class_id`.
  - **mAP@R** (Musgrave et al. 2020): para cada query, R = nº de itens da mesma
    classe na gallery; recupera top-R; `AP@R = (1/R) Σ_{i=1}^{R} [acerto_i] · P(i)`,
    onde `P(i)` = precisão acumulada até a posição i. `mAP@R` = média sobre queries.
  - Definir as fórmulas em docstring — mAP@R é fácil de errar.

### 5.5 `scripts/01_baseline_knn.py`
- Carrega embeddings cacheados → roda avaliação leave-one-out no split de teste →
  loga a linha de resultado (regime = `frozen_knn`) via `utils/logging.py`.
- **Saída esperada:** uma linha de métricas (Recall@{1,2,4,8}, mAP@R) para o CLIP
  congelado. Esse é o baseline.

---

## 6. LAYER 1 — Projection head treinada  *(detalhar bem)*

Objetivo: primeira contribuição própria de DL. Treinar uma cabeça sobre os
embeddings **congelados e cacheados** e medir o ganho sobre o baseline da Layer 0.
Treina rápido (CPU serve), porque o backbone não entra no grafo.

### 6.1 `src/models/projection.py`  *(exclusivo CM-204)*
- `ProjectionHead(in_dim=512, hidden_dim=512, out_dim=128)`:
  `Linear → BatchNorm1d → ReLU → Linear → (L2-normalize na saída)`.
- A saída L2-normalizada é o vetor usado tanto na perda quanto na recuperação.

### 6.2 `src/data/sampler.py`  *(exclusivo CM-204)*
- `PKSampler(labels, P, K)`: cada batch contém **P classes × K imagens** por classe
  (default P=8, K=4 → batch 32). Garante positivos e negativos em todo batch.
  Sampler customizado do PyTorch (agrupa índices por classe, sorteia P classes,
  depois K índices por classe). Defensável como implementação própria.

### 6.3 `src/mining/miners.py`  *(exclusivo CM-204)*
- Assinatura única com parâmetro `strategy`, projetada já pensando na Layer 2:
  - `strategy="batch_hard"` (**default da Layer 1**): para cada âncora, positivo
    mais distante + negativo mais próximo *dentro do batch* (Hermans et al. 2017).
  - `strategy="batch_all"`: média da triplet loss sobre **todos** os triplos
    válidos do batch. Custa quase nada e dá ablação interna grátis (triplet ingênua
    vs. minerada) — bom para o relatório.
- Layer 2 estende este módulo com `semi_hard` e `distance_weighted` (ver §7).

### 6.4 `src/losses/triplet.py`  *(exclusivo CM-204)*
- `TripletLoss(margin=0.2)` consumindo os índices selecionados pelo miner:
  `max(0, d(a,p) − d(a,n) + margin)`, distância euclidiana sobre vetores
  L2-normalizados.

### 6.5 `scripts/02_train_head.py`
- Carrega embeddings cacheados do **split de treino** (classes 1–100) → PK sampler →
  head → miner → triplet loss → otimização.
- **Hiperparâmetros default:** otimizador Adam, `lr=1e-3` (só a head treina),
  `epochs≈100` (barato sobre cache), `margin=0.2`, P=8, K=4, `seed` fixa.
- Ao fim de cada época (ou no fim): projeta o **split de teste** pela head treinada,
  roda a mesma avaliação leave-one-out da Layer 0, loga resultado.
- **Rodar com `batch_hard` e `batch_all`** → dois regimes a mais.

### 6.6 Entregável da Layer 1 (o que vai ao professor)
Uma tabela única, mesmas colunas para todos os regimes:

| Regime | R@1 | R@2 | R@4 | R@8 | mAP@R |
|--------|-----|-----|-----|-----|-------|
| `frozen_knn` (Layer 0) | … | … | … | … | … |
| `head_triplet_batch_all` (Layer 1) | … | … | … | … | … |
| `head_triplet_batch_hard` (Layer 1) | … | … | … | … | … |

Essa tabela é a munição da conversa com o professor: prova que o pipeline de treino
funciona end-to-end e que o desenho de variável única produz um delta interpretável.

---

## 7. LAYER 2 — Estudo de mineração  *(estrutura agora, implementação futura)*

Não entra no protótipo pré-26/06. A estrutura (`src/mining/miners.py` com parâmetro
`strategy`) já é criada de modo a acomodar isto sem refatorar:
- Adicionar `semi_hard` (negativos mais difíceis que o positivo, mas ainda dentro
  da margem) e `distance_weighted` (Wu et al. 2017).
- `scripts/04_mining_ablation.py`: roda os mesmos regimes variando só a estratégia
  de mineração → tabela comparativa. Eixo experimental "de graça" sobre a Layer 1.

---

## 8. LAYER 3 — Adaptação via LoRA  *(estrutura agora, implementação futura)*

Resultado central planejado e **deliverable núcleo** (carrega o critério de
complexidade). Entra **depois** da aprovação do tema; é o único regime que exige
GPU de verdade (Kaggle como ambiente principal de execução).
- `src/models/lora.py`: injeta adaptadores LoRA (Hu et al. 2021) nas projeções de
  atenção do ViT do CLIP; só os adaptadores + head treinam, resto congelado.
- Aqui o backbone **entra no grafo** → não dá mais para treinar só sobre cache;
  o forward passa pelas imagens. Por isso GPU.
- `scripts/05_train_lora.py`: mesmo protocolo de avaliação, novo regime
  `lora_triplet_batch_hard` na tabela. O contraste frozen vs. adaptado é o
  resultado central do artigo.

> **SupCon** (`src/losses/contrastive.py`) fica como **ablação de perda** documentada:
> mesmo sampler e head, troca só a perda, compara triplet vs. supervised-contrastive.
> Stub criado agora; implementação opcional no projeto completo.

---

## 9. Avaliação, logging e reprodutibilidade

- `src/utils/seed.py`: fixa seed de `random`, `numpy`, `torch` (+ determinismo cuDNN
  quando aplicável). Chamada no início de todo script.
- `src/utils/logging.py`: cada experimento grava **uma linha** em
  `results/experiments.csv` (e/ou JSON) com: timestamp, regime, backbone, perda,
  estratégia de mineração, hiperparâmetros (P, K, margin, lr, epochs, dim),
  flag de normalização, seed, e as métricas (R@1,2,4,8, mAP@R).
- Racional: a tabela do relatório IEEE sai direto deste arquivo, sem recompor
  números à mão. Atende ao critério de qualidade de código da disciplina.

---

## 10. README e requirements (desde o 1º commit, maleáveis)

`README.md` deve conter, desde o início (e crescer com o projeto):
- Identidade do projeto e a **pergunta de pesquisa** (§1).
- **Decisão de perda documentada explicitamente:** *triplet loss com batch-hard
  mining* foi a escolha primária — registrar o porquê (transparência de depuração
  para projeto solo, mineração como artefato de implementação própria, conexão com
  a literatura clássica de DML, e SupCon reservada como ablação). Você pediu para
  deixar isso claro no README; é relevante.
- **Fronteira de disclosure VIAL/CM-204** (§3).
- **Manual de execução:** instalar dependências (`pip install -r requirements.txt`),
  baixar o CUB, e a ordem dos scripts: `00 → 01 → 02 → 03`. Inclui que o professor
  consegue rodar localmente (CUB é leve; Layers 0–1 não exigem GPU).

`requirements.txt`: `torch`, `open_clip_torch`, `faiss-cpu`, `numpy`,
`datasets` (se usar mirror HF), `pyyaml`, `pandas`. Maleável — atualizar conforme
módulos forem adicionados.

---

## 11. Ordem de implementação (para o Claude Code)

Sugestão de sequência ao gerar o código, respeitando dependências:

1. Esqueleto: estrutura de pastas, `configs/default.yaml`, `utils/seed.py`,
   `utils/logging.py`, `.gitignore`, `README.md`, `requirements.txt`.
2. `data/cub.py` (carga + split metric-learning).
3. `models/backbone.py` + `cache/embeddings.py` + `scripts/00`.
4. `eval/retrieval.py` + `eval/metrics.py` + `scripts/01` → **fecha Layer 0**.
5. `models/projection.py` + `data/sampler.py` + `mining/miners.py` +
   `losses/triplet.py` + `scripts/02` + `scripts/03` → **fecha Layer 1**.
6. Stubs de `mining` (estratégias extras), `losses/contrastive.py`, `models/lora.py`,
   `scripts/04`, `scripts/05` → estrutura pronta para Layers 2 e 3.

**Meta do protótipo (pré-26/06):** passos 1–5 completos, produzindo a tabela do §6.6.
Passos 6 ficam como esqueleto descrito, não executado.

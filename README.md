# CM-204 — Metric Learning com CLIP congelado

## Identidade e pergunta de pesquisa

Estudo experimental de **metric learning** sobre um backbone **CLIP congelado**,
medindo o ganho de adicionar treino (cabeça de projeção e, no futuro, LoRA)
sobre a representação congelada, na tarefa de **recuperação de imagens em
nível de instância** no dataset **CUB-200-2011**.

O eixo experimental é um protocolo de **variável única**: a representação
muda de regime (congelada → head treinada → backbone adaptado via LoRA)
enquanto dataset, split, métricas e protocolo de avaliação permanecem fixos.
Cada regime produz a mesma tabela de métricas, tornando o ganho de cada
incremento diretamente legível.

> **Nota de escopo.** No TG existe uma comparação CLIP vs. DINOv2. Neste
> projeto usamos **somente CLIP** — nenhum código de DINOv2 entra aqui.

Detalhes completos de decisões, contratos de módulo e racional ficam em
[SPEC_CM204.md](SPEC_CM204.md) — este README resume o necessário para rodar e
entender o projeto.

## Decisão de perda

A escolha primária é **triplet loss com batch-hard mining** (Hermans et al.,
2017), em vez de, por exemplo, supervised contrastive (SupCon) como perda
principal. Motivos, registrados explicitamente:

- **Transparência de depuração** em projeto solo: triplet loss com mineração
  explícita deixa visível, a cada batch, qual foi o positivo/negativo
  escolhido — mais fácil de auditar que uma perda contrastiva agregada.
- **Mineração como artefato de implementação própria**: o módulo
  `src/mining/miners.py` é, junto com a head e o PK sampler, parte da
  contribuição de DL exclusiva deste projeto (ver seção de disclosure abaixo).
- **Conexão com a literatura clássica de DML** (deep metric learning), onde
  triplet + mineração é o ponto de partida canônico.
- **`batch_all`** é mantida como ablação interna praticamente gratuita
  (triplet ingênua vs. minerada), rodada junto com `batch_hard` no mesmo
  script.
- **SupCon** (`src/losses/contrastive.py`) fica reservada como ablação de
  perda futura — stub criado, implementação fora do escopo do protótipo
  atual.

## Fronteira de disclosure (VIAL / CM-204)

- **Herdado do VIAL** (contexto temático + infraestrutura de recuperação): a
  pergunta de pesquisa e a camada de *retrieval* (índice FAISS, busca kNN,
  métricas de recuperação). Concentrado em `src/eval/`.
- **Exclusivo de CM-204** (contribuição própria — onde mora a nota de
  código): todo o pipeline de treino — cabeça de projeção, triplet loss,
  mineração de negativos, PK sampler e (futuro) LoRA. Concentrado em
  `src/models/projection.py`, `src/losses/`, `src/mining/`,
  `src/models/lora.py`.

> *"O pipeline de treinamento (projection head, triplet loss, mineração de
> negativos e adaptação via LoRA) foi implementado exclusivamente para
> CM-204. Do contexto do VIAL herdam-se apenas a motivação temática e a
> infraestrutura de embeddings/recuperação."*

Os cabeçalhos de cada arquivo em `src/` trazem as tags `[CM-204]` ou
`[VIAL-infra]` reforçando essa fronteira.

## Estrutura do projeto

Ver a árvore completa e o contrato de cada módulo em
[SPEC_CM204.md §4](SPEC_CM204.md). Resumo das camadas (*layers*):

- **Layer 0** — baseline congelado + kNN (`scripts/00`, `scripts/01`).
- **Layer 1** — projection head treinada com triplet loss (`scripts/02`,
  `scripts/03`).
- **Layer 2** — ablação de mineração (`scripts/04`) — esqueleto, não
  implementada.
- **Layer 3** — adaptação via LoRA (`scripts/05`) — esqueleto, não
  implementada; resultado central planejado do projeto completo.

## Manual de execução

```bash
pip install -r requirements.txt
```

### Dados

Baixe e extraia o `CUB_200_2011.tgz` oficial (Caltech) em
`data/CUB_200_2011/` (estrutura padrão: `images/`, `images.txt`,
`image_class_labels.txt`, `classes.txt`). Esse é o caminho **local** — único
default, fixo e plenamente suportado (`configs/default.yaml`,
`data.source: "local"`): reproduzível, determinístico, sem depender de um
mirror de terceiros. `train_test_split.txt` é ignorado de propósito — é o
split de classificação original do CUB, e o split deste estudo é por classe
(1-100 treino, 101-200 teste), não por imagem.

Há também um caminho `"hf"` (mirror no Hugging Face Hub via `datasets`)
wired em `src/data/cub.py` como **fallback opcional, não-default**: o
dataset exato e o schema de campos **não foram confirmados** — confirme em
`configs/default.yaml` (`data.hf_dataset_name`, `hf_image_field`,
`hf_label_field`) antes de usá-lo.

### Ordem dos scripts

```bash
python scripts/00_cache_embeddings.py   # extrai e cacheia embeddings do CLIP (Layer 0)
python scripts/01_baseline_knn.py       # baseline congelado + kNN (Layer 0)
python scripts/02_train_head.py         # treina a head (batch_hard e batch_all) (Layer 1)
python scripts/03_eval.py               # consolida results/experiments.csv na tabela final
```

As Layers 0-1 não exigem GPU: a extração de embeddings (`00`) é a única etapa
que se beneficia de acelerar (Colab/Kaggle) caso fique lenta em CPU; tudo
depois roda sobre os embeddings cacheados.

O loop de treino da head (`src/training/head_trainer.py:train_head`) é
chamado uma vez por estratégia de mineração e foi desenhado para ser
reusado, sem refatorar, pela Layer 2 (`scripts/04_mining_ablation.py`)
quando `semi_hard`/`distance_weighted` forem implementadas.

## Normalização (cache raw, decisão por consumidor)

`src/models/backbone.py:ClipBackbone.encode` **nunca normaliza** — o cache em
`data/embeddings/` guarda sempre o embedding raw do CLIP. A normalização é
decisão de cada consumidor:

- **Baseline kNN** (`scripts/01_baseline_knn.py`): aplica
  `retrieval.baseline_l2_normalize` (default `True`) sobre os embeddings raw
  antes da busca — flipar essa flag para a ablação do §2.1 não exige
  recachear nada.
- **Projection head** (Layer 1): recebe o embedding raw como entrada (o
  `BatchNorm1d` da head absorve a escala); a saída é sempre L2-normalizada
  por construção, independentemente da flag do baseline.

## Reprodutibilidade e logging

Toda execução fixa a seed global (`src/utils/seed.py`) e grava uma linha de
resultado em `results/experiments.csv` / `results/experiments.json`
(`src/utils/logging.py`), com hiperparâmetros e métricas completas. A tabela
do relatório IEEE é montada diretamente desse arquivo via `scripts/03_eval.py`,
sem recompor números à mão.

Durante o treino da head, cada checkpoint de avaliação (a cada
`training.eval_every` épocas) também é gravado em
`results/training_head_triplet_<strategy>.csv` — a curva de convergência
(época × métrica), útil para a análise ligada à teoria no relatório.

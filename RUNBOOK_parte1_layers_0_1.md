# RUNBOOK — Parte 1 · Layers 0 e 1

> **Escopo.** Sair de "código commitado, nunca executado" para **números reais na tabela**.
> Cobre: setup do ambiente, obtenção do CUB-200-2011, execução da Layer 0 (baseline
> congelado + kNN) e da Layer 1 (cabeça de projeção treinada).
>
> **Estado de partida.** Os 37 arquivos estão commitados em `CM204/` (branch `main`).
> `torch`, `open_clip` e `datasets` **não** estavam instalados no ambiente de escrita —
> o pipeline `00 → 03` **nunca rodou ponta a ponta**.
>
> **Expectativa realista.** A primeira execução é uma **sessão de depuração**, não uma
> formalidade. Descasamentos de shape, nome de campo e indexação FAISS são normais no
> primeiro contato com dados reais. Reserve tempo para isso.

---

## 1. Setup do ambiente (Cursor — caminho principal)

Todo o trabalho das Layers 0 e 1 roda **em CPU**, porque a head treina sobre embeddings
cacheados. A única etapa que se beneficia de GPU é a extração inicial (§3).

### 1.1 Ambiente virtual

Abrir o terminal integrado do Cursor na raiz do projeto (`CM204/`).

**Windows (PowerShell)** — seu caso:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

> Se o PowerShell bloquear o script de ativação:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Depois, no Cursor: `Ctrl+Shift+P` → *Python: Select Interpreter* → escolher `.venv`.

### 1.2 Instalar dependências

`torch` é a única que pede atenção — o `pip install torch` puro pode baixar o build
CUDA (~2.5 GB) sem necessidade. Para Layers 0 e 1, **CPU basta**:

```bash
# PyTorch CPU-only (menor e suficiente para Layers 0-1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# O resto do projeto
pip install -r requirements.txt
```

> Se você já tem GPU NVIDIA local e quer usá-la desde já, troque a primeira linha por
> `pip install torch torchvision` (build CUDA default). Não é necessário agora.

### 1.3 Sanity check das importações

```bash
python -c "import torch, open_clip, faiss, numpy, pandas; print('ok', torch.__version__)"
```

Se `faiss-cpu` falhar no Windows (acontece), a alternativa é adiar: nesta escala
(~5.8k vetores) uma busca exata em `torch.cdist` é equivalente. Mas tente primeiro —
manter FAISS preserva a coerência com a infra herdada do VIAL.

---

## 2. Obter o CUB-200-2011

**Você não tira nenhuma foto.** É um dataset acadêmico pronto: 11.788 imagens,
200 espécies, com rótulo de classe por imagem. Baixa-se uma vez.

### 2.1 Fontes (em ordem de preferência)

1. **CaltechDATA (canônica, DOI `10.22002/D1.20098`):**
   `https://data.caltech.edu/records/65de6-vp158`
   Baixar o `CUB_200_2011.tgz` pela página.
2. **URL histórica** (frequentemente fora do ar, tentar só se conveniente):
   `http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz`
3. **Mirror Git LFS** (fallback):
   `https://media.githubusercontent.com/media/vignagajan/CUB-200-2011/main/CUB_200_2011.tgz`

### 2.2 Baixar e extrair

**Linux / macOS / Colab:**
```bash
mkdir -p data/raw && cd data/raw
wget -O CUB_200_2011.tgz "<URL_ESCOLHIDA>"
tar -xzf CUB_200_2011.tgz
cd ../..
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Force -Path data\raw | Out-Null
# Baixe o .tgz pelo navegador (CaltechDATA) e mova para data\raw\
tar -xzf data\raw\CUB_200_2011.tgz -C data\raw
```
> `tar` já vem no Windows 10+. Não precisa de 7-Zip.

### 2.3 Estrutura esperada (o loader depende disso)

```
data/raw/CUB_200_2011/
├── images/                     # 200 subpastas, uma por espécie
├── images.txt                  # image_id → caminho relativo
├── image_class_labels.txt      # image_id → class_id (1..200)
├── classes.txt                 # class_id → nome da espécie
└── train_test_split.txt        # IGNORADO (nosso split é por classe)
```

Verificação rápida:
```bash
wc -l data/raw/CUB_200_2011/images.txt          # deve dar 11788
wc -l data/raw/CUB_200_2011/classes.txt         # deve dar 200
```
(No PowerShell: `(Get-Content ... | Measure-Object -Line).Lines`)

### 2.4 Apontar o config

Em `configs/default.yaml`, confirmar:
```yaml
data:
  source: "local"
  root: "data/raw/CUB_200_2011"
```

---

## 3. Extração de embeddings (script `00`)

Passa **todas** as 11.788 imagens pelo CLIP ViT-B/16 congelado, **uma única vez**, e
salva um tensor `[N, 512]` **raw (não normalizado)** com arrays paralelos de
`class_id` e `image_id`.

```bash
python scripts/00_cache_embeddings.py --config configs/default.yaml
```

**Custo.** Em CPU, esperar algo entre dezenas de minutos e ~2h (depende da máquina).
É uma passada única de inferência — depois disso, **nada mais precisa de GPU nas
Layers 0 e 1**.

### 3.1 Se quiser acelerar no Colab (opcional)

Vale a pena só se a CPU estiver muito lenta. A estratégia é: **GPU só executa, o
código continua morando no git.**

```python
# Célula 1 — GPU: Runtime > Change runtime type > T4 GPU
!nvidia-smi

# Célula 2 — clonar o repo
!git clone https://github.com/<seu-usuario>/<seu-repo>.git
%cd <seu-repo>

# Célula 3 — dependências (Colab já traz torch com CUDA)
!pip install -q open_clip_torch faiss-cpu pyyaml pandas tabulate tqdm Pillow

# Célula 4 — baixar o CUB direto no Colab
!mkdir -p data/raw && cd data/raw && wget -q -O CUB_200_2011.tgz "<URL>" && tar -xzf CUB_200_2011.tgz

# Célula 5 — extrair embeddings (minutos, não horas)
!python scripts/00_cache_embeddings.py --config configs/default.yaml

# Célula 6 — baixar o cache de volta para a máquina local
from google.colab import files
files.download('data/embeddings/clip_vitb16.npy')
```

Coloque o `.npy` em `data/embeddings/` local e siga daqui em CPU. O cache é o
artefato portátil — todo o resto das Layers 0 e 1 roda sobre ele.

> **Se o repo for privado:** clone com token
> (`git clone https://<TOKEN>@github.com/...`), ou simplesmente faça upload de um zip
> do projeto pela barra lateral do Colab. Para uma execução única, o zip é mais simples.

---

## 4. LAYER 0 — Baseline congelado + kNN (script `01`)

Objetivo: o **piso de referência**, sem treino nenhum. É o número contra o qual todo
ganho posterior é medido, e valida que dados, backbone e avaliação funcionam.

```bash
python scripts/01_baseline_knn.py --config configs/default.yaml
```

**O que acontece internamente:**
- Carrega o cache raw, filtra o **split de teste** (classes 101–200).
- Aplica L2-norm conforme `retrieval.baseline_l2_normalize` (**na avaliação**, não no cache).
- Constrói o índice FAISS e busca os vizinhos em protocolo **leave-one-out**: cada
  embedding de teste é query; a gallery é o restante do split (excluindo a própria query).
- Calcula Recall@{1,2,4,8} e mAP@R.
- Grava **uma linha** em `results/experiments.csv` com regime `frozen_knn`.

### 4.1 O que checar (não pule)

- `class_id` do split de teste está no intervalo **101–200**, e nenhuma classe de treino
  vazou. Zero interseção — é a espinha dorsal do protocolo.
- A query **não** aparece entre os próprios vizinhos (leave-one-out correto). Se o R@1
  vier suspeito de perfeito (≈1.0), é exatamente esse o bug.
- Ordem de grandeza sanity: para CLIP ViT-B congelado no CUB, um R@1 na casa dos
  ~50–60% é plausível. Se der ~1% (aleatório) ou ~100%, há bug — não é resultado.

### 4.2 Ablação grátis

Rodar de novo com `baseline_l2_normalize: false` gera uma segunda linha e um ponto de
discussão para o relatório (efeito da normalização no espaço de busca). Como o cache é
raw, **não precisa recachear** — foi exatamente para isso que desacoplamos.

---

## 5. LAYER 1 — Cabeça de projeção treinada (script `02`)

Primeira contribuição própria de DL. Treina um MLP sobre os embeddings **congelados e
cacheados** e mede o ganho sobre a Layer 0. O backbone não entra no grafo, então roda
rápido em CPU.

```bash
python scripts/02_train_head.py --config configs/default.yaml
```

**O que acontece internamente:**
- Carrega o cache raw do **split de treino** (classes 1–100).
- `PKSampler` monta batches de P=8 classes × K=4 imagens (batch 32), garantindo
  positivos e negativos em todo batch.
- `ProjectionHead` (512 → BatchNorm/ReLU → 128, saída **sempre L2-normalizada**) recebe
  o embedding **raw** como input (o BatchNorm cuida da escala).
- O miner seleciona triplos conforme `strategy`; a `TripletLoss` (margem 0.2) otimiza.
- `train_head()` é chamado **uma vez por estratégia**, com **seed resetada antes de cada**
  — cada head é independentemente reproduzível.
- A cada `eval_every` épocas, projeta o split de teste, roda a **mesma** avaliação
  leave-one-out da Layer 0 e persiste em `results/training_head_triplet_<strategy>.csv`.
- No fim, grava a linha final em `results/experiments.csv`.

### 5.1 Hiperparâmetros default

| Parâmetro | Valor |
|---|---|
| Otimizador | Adam, `lr=1e-3` (só a head treina) |
| Épocas | ~100 (barato sobre cache) |
| Margem (triplet) | 0.2 |
| P × K | 8 × 4 (batch 32) |
| Saída da head | 128-d, L2-normalizada |
| Estratégias | `batch_all` e `batch_hard` |

### 5.2 O que checar

- **`batch_hard` colapsando?** Se todos os embeddings convergirem para um ponto (loss
  cai a zero e as métricas desabam), é o modo de falha clássico da mineração agressiva.
  Mitigações: mais épocas de *warm-up* com `batch_all` antes de trocar, `lr` menor, ou
  margem um pouco maior.
- **`batch_all` estagnando?** Esperado até certo ponto — a maioria dos triplos é fácil e
  tem gradiente zero. É justamente o contraste que você quer mostrar no relatório.
- A **curva de convergência** persistida é figura pronta para o artigo. Não a descarte.

---

## 6. Tabela consolidada (script `03`)

```bash
python scripts/03_eval.py --config configs/default.yaml
```

Lê `results/experiments.csv` e emite a tabela em markdown (via `tabulate`), pronta para
o relatório:

| Regime | R@1 | R@2 | R@4 | R@8 | mAP@R |
|--------|-----|-----|-----|-----|-------|
| `frozen_knn` (Layer 0) | … | … | … | … | … |
| `head_triplet_batch_all` (Layer 1) | … | … | … | … | … |
| `head_triplet_batch_hard` (Layer 1) | … | … | … | … | … |

**Leitura esperada pela teoria:** a head treinada deve superar o congelado (o espaço
CLIP é semanticamente enviesado, e o metric learning o reorganiza para discriminação em
nível de instância); e `batch_hard` deve superar `batch_all` (triplos fáceis não
carregam gradiente). Se algo contrariar isso, **não esconda** — explicar uma anomalia
com a teoria vale mais nota do que um número bonito sem análise.

---

## 7. Checklist de conclusão da Parte 1

- [ ] `.venv` ativo, `import torch, open_clip, faiss` sem erro
- [ ] `data/raw/CUB_200_2011/` com `images.txt` (11788 linhas) e `classes.txt` (200)
- [ ] `scripts/00` gerou `data/embeddings/clip_vitb16.npy` (raw, `[11788, 512]`)
- [ ] `scripts/01` gravou a linha `frozen_knn` — R@1 em ordem de grandeza plausível
- [ ] Split de teste verificado: só classes 101–200, zero interseção com treino
- [ ] Leave-one-out verificado: query não é seu próprio vizinho
- [ ] `scripts/02` gravou as duas linhas de head + as duas curvas de treino
- [ ] `scripts/03` emite a tabela em markdown
- [ ] `git commit` com `results/experiments.csv` versionado (é pequeno e é evidência)

Com o checklist fechado, você tem **números reais** — e pode começar a escrever as
seções de resultado do relatório enquanto ataca a Parte 2.

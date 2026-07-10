# RUNBOOK — Parte 2 · Layers 2 e 3

> **Escopo.** O que vem **depois** de a Parte 1 produzir números reais.
> Layer 2: aprofundar o estudo de mineração (`semi_hard`, `distance_weighted`).
> Layer 3: adaptação do backbone via **LoRA** — o deliverable central do projeto,
> onde a GPU entra de verdade.
>
> **Não implementar agora.** Este documento é a especificação para quando a Parte 1
> estiver fechada. A estrutura de arquivos já existe e acomoda tudo isto sem refatorar.

---

## PARTE A — LAYER 2 · Estudo de mineração

### A.1 Por que esta layer existe

`batch_hard` já entrou na Layer 1, então a Layer 2 **não introduz mineração** — ela
transforma a mineração num **eixo experimental**. Mantendo dataset, split, head, perda,
sampler e protocolo de avaliação fixos, varia-se **apenas a estratégia de seleção de
triplos**. É o protocolo de variável única aplicado ao componente mais sensível do
metric learning.

O valor no relatório: `batch_hard` é conhecido por ser instável — ele persegue os
negativos mais próximos, que muitas vezes são *outliers* ou imagens mal rotuladas, e
isso pode colapsar o embedding. As duas estratégias abaixo são as respostas clássicas
da literatura a esse problema, e comparar as quatro (`batch_all`, `batch_hard`,
`semi_hard`, `distance_weighted`) é exatamente a "análise aprofundada ligada à teoria"
que a disciplina cobra.

### A.2 `semi_hard` (FaceNet — Schroff et al., 2015)

**Ideia.** Não pegue o negativo mais próximo (arriscado); pegue um negativo que já é
"difícil" mas ainda não é *mais próximo que o positivo*. Formalmente, para uma âncora
`a` com positivo `p`, um negativo `n` é semi-hard quando:

```
d(a, p)  <  d(a, n)  <  d(a, p) + margin
```

Ou seja: ele está **fora** do positivo (o triplo já está "correto"), mas **dentro** da
margem (ainda gera gradiente). Isso evita tanto os triplos triviais (`d(a,n)` enorme,
gradiente zero) quanto os triplos degenerados (`d(a,n) < d(a,p)`, que puxam o modelo
para o colapso).

**Implementação em `src/mining/miners.py`:**
- Para cada par `(a, p)` válido do batch, montar a máscara de negativos que satisfazem
  a dupla desigualdade acima.
- **Caso de borda crítico:** se nenhum negativo for semi-hard para aquele par (comum no
  fim do treino), há duas políticas — (i) descartar o par, ou (ii) cair para o negativo
  mais difícil disponível. Escolha uma, **documente no README**, e seja consistente. A
  política (ii) é mais robusta; a (i) é mais fiel ao paper.
- Selecionar aleatoriamente entre os semi-hard válidos (o FaceNet original faz isso).

### A.3 `distance_weighted` (Wu et al., 2017 — "Sampling Matters")

**Ideia.** Em vez de escolher determinística ou uniformemente, **amostre** o negativo
com probabilidade inversamente proporcional à densidade de pares àquela distância.

A motivação é geométrica e vale um parágrafo no relatório: em alta dimensão, sobre a
hiperesfera unitária, as distâncias par-a-par se concentram fortemente em torno de
`√2`. Amostragem uniforme, portanto, quase só devolve negativos daquela faixa — pouco
informativos. A amostragem ponderada por distância corrige esse viés, produzindo um
espectro de dificuldades e um gradiente com **variância muito menor** que a do
`batch_hard`.

**Implementação:**
- Peso `∝ q(d)^{-1}`, onde `q(d)` é a densidade analítica das distâncias par-a-par
  numa esfera de dimensão `n` (a fórmula está no paper; a head projeta em 128-d).
- Na prática: computar as distâncias `d(a, n)` no batch, transformar em log-pesos,
  **clampar** (as distâncias muito pequenas explodem o peso), aplicar softmax e amostrar.
- Costuma vir com um **cutoff inferior** (ex. `d < 0.5` é truncado) para estabilidade.
- Este é o mais delicado dos quatro — a clampagem numérica é onde se erra.

### A.4 Execução

`src/training/head_trainer.py` já expõe `train_head(strategy, seed, ...)`, e o
`04_mining_ablation.py` está documentado para reusá-la. Não há refatoração:

```bash
python scripts/04_mining_ablation.py --config configs/default.yaml
```

Iterar sobre `["batch_all", "batch_hard", "semi_hard", "distance_weighted"]`, resetando
a seed antes de cada. Saída: quatro linhas em `experiments.csv` e quatro curvas de
treino persistidas.

**Entregável:** tabela comparativa + figura sobrepondo as quatro curvas de convergência.
A figura é forte — ela mostra visualmente a instabilidade do `batch_hard` contra a
suavidade do `distance_weighted`, se a teoria se confirmar.

**Custo:** baixo. Roda em CPU sobre o cache, como a Layer 1. Sem GPU.

---

## PARTE B — LAYER 3 · Adaptação via LoRA

### B.1 A mudança estrutural (leia antes de codar)

Até aqui, **o backbone nunca entrou no grafo computacional**. As Layers 0, 1 e 2 rodam
inteiramente sobre `data/embeddings/clip_vitb16.npy` — vetores pré-computados. Por isso
rodavam em CPU.

Na Layer 3 isso acaba. O LoRA modifica os pesos do ViT, então **os embeddings mudam a
cada passo de otimização** e o cache torna-se inútil. O forward passa pelas **imagens**.

Três consequências concretas, e a terceira é a que pega as pessoas:

1. **GPU deixa de ser opcional.** Backprop através de um ViT-B/16 em batches de 32
   imagens 224×224 não é viável em CPU.
2. **O `DataLoader` muda de natureza.** Ele passa a servir tensores de imagem
   pré-processados, não vetores. O `PKSampler` continua igual (ele opera sobre
   índices/labels), mas o `Dataset` por trás dele agora abre arquivos com Pillow.
3. **O caminho de carregamento de imagem, ocioso até agora, vira load-bearing.** O
   `transform` do `open_clip` (`create_model_and_transforms` devolve `preprocess`) tem
   que ser aplicado corretamente — resize, center-crop, normalização com a média/desvio
   do CLIP. **Um erro silencioso aqui degrada tudo sem levantar exceção.** Valide cedo:
   passe uma imagem pelo `preprocess` + backbone congelado e confirme que o embedding
   bate com o que está no cache da Layer 0. Se não bater, o pré-processamento está errado.

### B.2 `src/models/lora.py`

**Teoria (Hu et al., 2021).** Em vez de atualizar `W ∈ ℝ^{d×k}`, congela-se `W` e
aprende-se um resíduo de baixo posto: `W' = W + (α/r)·BA`, com `B ∈ ℝ^{d×r}`,
`A ∈ ℝ^{r×k}` e `r ≪ min(d,k)`. `A` inicializa gaussiano, `B` inicializa **zero** — de
modo que no passo 0 o modelo é *exatamente* o pré-treinado, e a adaptação parte da
identidade. Só `A` e `B` recebem gradiente.

**Onde injetar.** Nas projeções de atenção (`q_proj`, `v_proj` são o padrão do paper;
injetar em `q` e `v` costuma bastar) dos blocos do ViT visual do CLIP.

**Hiperparâmetros de partida:** `r ∈ {4, 8, 16}`, `α = 2r` (regra prática comum),
`dropout ≈ 0.1`. Um `r` pequeno já entrega a maior parte do ganho — o paper de LoRA é
enfático nisso, e vale reportar um pequeno *sweep* de `r` como ablação.

**Duas taxas de aprendizado.** A head e os adaptadores têm escalas de gradiente
diferentes. Use *param groups*: `lr` menor para os adaptadores (ex. `1e-4`) e maior para
a head (ex. `1e-3`). Treinar tudo com o mesmo `lr` é um erro comum.

**O que treina:** adaptadores LoRA + `ProjectionHead`. Todo o resto congelado. Logue a
contagem de parâmetros treináveis vs. totais — é um número que **impressiona no
relatório** e é o argumento central de PEFT (tipicamente <1%).

### B.3 `scripts/05_train_lora.py`

Mesmo protocolo de avaliação, mesma tabela, novo regime `lora_triplet_batch_hard`. A
avaliação continua leave-one-out sobre o split de teste — mas agora exige um forward do
backbone adaptado sobre as imagens de teste (não há cache).

**Fixar a estratégia de mineração** (use a vencedora da Layer 2) para que a única
variável entre Layer 1 e Layer 3 seja *o backbone estar adaptado ou não*. É o coração
do protocolo de variável única, e o resultado central do artigo.

---

## PARTE C — Transportar o código para GPU (Kaggle / Colab)

**Princípio.** O código **mora no git**; o ambiente de GPU apenas **executa**. Nunca
edite código dentro do notebook — cole-o e você perde versionamento, reprodutibilidade e
a nota de qualidade de código.

### C.1 Kaggle (ambiente principal — ~30h/semana, background execution)

O diferencial do Kaggle é o **background execution**: o treino continua mesmo com o
navegador fechado. Para LoRA (horas de treino), é decisivo.

**Passo 1 — subir o CUB como Kaggle Dataset (uma vez).** Não baixe o `.tgz` a cada
sessão. *Datasets → New Dataset → Upload* o `CUB_200_2011.tgz` (~1.1 GB). Ele fica
montado em `/kaggle/input/<nome-do-dataset>/` e persiste entre sessões, read-only.

> Alternativa: já existem mirrors públicos do CUB no Kaggle. Procure e anexe via
> *Add Data* — evita o upload.

**Passo 2 — o notebook.** *New Notebook* → *Settings* → *Accelerator: GPU T4 x2* (ou
P100) → *Internet: On* (necessário para `git clone` e `pip install`).

```python
# Célula 1 — sanity
!nvidia-smi

# Célula 2 — código vem do git, não do notebook
!git clone https://github.com/<usuario>/<repo>.git /kaggle/working/cm204
%cd /kaggle/working/cm204

# Célula 3 — deps (Kaggle já traz torch com CUDA; não reinstale torch)
!pip install -q open_clip_torch faiss-cpu pyyaml pandas tabulate tqdm

# Célula 4 — apontar o config para o dataset montado
!sed -i 's|root: .*|root: /kaggle/input/<nome-do-dataset>/CUB_200_2011|' configs/default.yaml

# Célula 5 — treino LoRA (aqui a GPU trabalha)
!python scripts/05_train_lora.py --config configs/default.yaml

# Célula 6 — resultados persistem em /kaggle/working, baixáveis pela aba Output
!cat results/experiments.csv
```

**Passo 3 — background execution.** *Save Version → Save & Run All (Commit)*. O
notebook roda do início ao fim no servidor; feche o navegador. Ao terminar, os arquivos
de `/kaggle/working/` ficam disponíveis na aba *Output* da versão.

> **Limites que importam:** sessão de GPU tem teto de ~9h (mais que suficiente para
> LoRA no CUB) e a cota semanal é de ~30h. `/kaggle/working/` (o que persiste) tem
> ~20 GB. Não escreva checkpoints gigantes — só os adaptadores LoRA e a head, que são
> pequenos (é a graça do PEFT).

**Passo 4 — trazer os resultados de volta.** Baixe `results/experiments.csv` e as
curvas de treino pela aba *Output*, e commite-os no repo local. **Os resultados devem
estar versionados no git**, não presos no Kaggle — o professor precisa vê-los, e o
`.zip` de entrega sai do repo.

### C.2 Colab (complemento — prototipagem rápida)

Melhor para *iterar* (depurar o `lora.py`, checar que o forward roda, medir tempo por
época) do que para treinar longo. A sessão cai por inatividade e o disco é efêmero.

```python
!nvidia-smi
!git clone https://github.com/<usuario>/<repo>.git && cd <repo>
%cd <repo>
!pip install -q open_clip_torch faiss-cpu pyyaml tabulate

from google.colab import drive           # persistir dataset e checkpoints
drive.mount('/content/drive')
# copie o CUB extraído do Drive, ou baixe uma vez e salve lá

!python scripts/05_train_lora.py --config configs/default.yaml --epochs 2   # smoke test
```

**Padrão recomendado:** Colab para um *smoke test* de 2 épocas (o LoRA converge? a loss
cai? quanto tempo por época?), Kaggle para a corrida completa em background.

### C.3 Higiene de repositório

Antes de clonar em qualquer ambiente de GPU, confirme que o `.gitignore` cobre `data/`,
`data/embeddings/` e checkpoints — caso contrário um `git add .` distraído tenta subir
1.1 GB de pássaros. Commite **só** código, configs, e os CSVs de resultado (pequenos e
que são a evidência do trabalho).

---

## PARTE D — Ablações opcionais e o relatório

Com Layers 0–3 fechadas, o núcleo do artigo existe. Extras, em ordem de custo-benefício:

- **Sweep de `r` no LoRA** (`r ∈ {4, 8, 16}`): barato, conecta direto com a tese de PEFT
  de que posto baixo basta. Alta relação valor/esforço.
- **SupCon vs. triplet** (`src/losses/contrastive.py`, hoje stub): mesmo sampler, mesma
  head, troca só a perda. Eixo de comparação limpo.
- **Stanford Online Products (stretch):** só se sobrar tempo *e* recurso. É um dataset
  muito maior — o custo de extração e treino não é comparável ao do CUB.

### D.1 O que dá para escrever **antes** de qualquer resultado

Aproveite o tempo de treino do LoRA (horas) para redigir as seções que não dependem de
número: a **introdução teórica** (viés semântico do CLIP vs. discriminação em nível de
instância; metric learning; triplet loss e o problema dos triplos fáceis; mineração;
LoRA e PEFT) e a **explicação da implementação**, escrita olhando o código commitado.
São, com folga, metade das oito páginas.

O que **não** dá para adiantar é *Resultados e Discussão* — e é lá que mora a análise
aprofundada que vale metade da nota. Por isso a sequência: rodar Layers 0 e 1 (rápido,
CPU), escrever teoria e implementação enquanto o LoRA treina, e fechar com os resultados.

### D.2 Lembretes de entrega

- Relatório **IEEE, máximo 8 páginas**, com Introdução Teórica, Implementação,
  Resultados/Discussões e Conclusão.
- **Manual do usuário** + `requirements.txt` fiel. O professor precisa conseguir rodar —
  as Layers 0 e 1 rodam na máquina dele, em CPU. Deixe isso explícito no README.
- **Disclosure VIAL/CM-204** explícito no relatório: o pipeline de treino (head, triplet,
  mineração, LoRA) é exclusivo de CM-204; do VIAL herdam-se motivação temática e a
  infraestrutura de embeddings/recuperação.
- Empacotar como `grupoX_exame.zip` e entregar pelo Google Classroom.

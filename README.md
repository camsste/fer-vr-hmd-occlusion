# Facial Expression Recognition under Simulated VR HMD Occlusion

Este repositório contém o código desenvolvido para avaliar o impacto da oclusão causada por Head-Mounted Displays (HMDs) de Realidade Virtual no reconhecimento automático de expressões faciais.

O experimento compara dois modelos de Facial Expression Recognition (FER), **POSTER++** e **QCS**, utilizando os datasets **RAF-DB** e **AffectNet+** sob quatro cenários de visualização facial:

- Baseline;
- HMD;
- HMD-Eyes;
- Real HMD-Eyes.

O objetivo é analisar como a oclusão da região superior da face, especialmente olhos e sobrancelhas, afeta o desempenho dos modelos e verificar se a inserção de informações visuais na região ocluída pode reduzir essa perda de desempenho.

---

## Visão geral

Modelos de reconhecimento de expressões faciais normalmente utilizam informações visuais de toda a face, incluindo olhos, sobrancelhas, testa, boca e relações espaciais entre essas regiões.

No entanto, em aplicações de Realidade Virtual, o uso de headsets cobre uma parte importante da região superior da face. Para investigar esse impacto de forma controlada, este projeto gera diferentes versões das mesmas imagens faciais e avalia os modelos sob condições equivalentes.

O pipeline desenvolvido realiza:

1. organização das imagens originais dos datasets;
2. detecção de landmarks faciais;
3. geração dos cenários de oclusão;
4. inserção de olhos geométricos e olhos mais realistas;
5. execução dos modelos POSTER++ e QCS;
6. cálculo de métricas globais e por classe;
7. geração de relatórios e matrizes de confusão normalizadas.

---

## Cenários experimentais

Para cada imagem, são geradas quatro versões.

### 1. Baseline

Imagem facial original, sem nenhuma modificação.

```text
Baseline
```

Esse cenário é utilizado como referência para comparar o desempenho dos modelos quando toda a face está visível.

### 2. HMD

Uma oclusão sintética é aplicada na região superior da face, simulando a cobertura causada por um headset de Realidade Virtual.

```text
HMD
```

A oclusão cobre principalmente a testa, sobrancelhas e parte da região dos olhos.

### 3. HMD-Eyes

A oclusão do headset é mantida, mas olhos geométricos são inseridos na região coberta.

```text
HMD_Olhos
```

Nesse cenário, são utilizados elementos visuais simples para representar os olhos, incluindo formas geométricas para esclera e íris.

### 4. Real HMD-Eyes

A oclusão do headset também é mantida, mas são inseridas texturas de olhos mais realistas.

```text
HMD_Olhos_Reais
```

As texturas são ajustadas de acordo com a posição, escala e inclinação estimadas a partir dos landmarks faciais.

Os arquivos utilizados como overlays de olhos são:

```text
Olho_Direito.png
Olho_Esquerdo.png
```

---

## Modelos avaliados

Dois modelos de reconhecimento de expressões faciais foram avaliados sob as mesmas condições experimentais.

### POSTERis modelos de reconhecimento de expressões faciais foram avaliados sob as mesmas condições experimentais.

### POSTER++

POSTER++ é um modelo de reconhecimento de expressões faciais baseado em transformadores e fusão de informações faciais.

A implementação e os scripts relacionados estão localizados em:

```text
POSTER_V2/
```

### QCS

QCS, ou Quadruplet Cross Similarity, utiliza uma estratégia de refinamento de características faciais baseada em similaridade entre exemplos.

A implementação e os scripts relacionados estão localizados em:

```text
QCS/
```

Os dois modelos são executados separadamente, pois possuem dependências, pesos e configurações específicas.

---

## Datasets

Os experimentos utilizam dois datasets de reconhecimento de expressões faciais em condições naturais.

### RAF-DB

O RAF-DB é utilizado com sete classes emocionais:

```text
Surprise
Fear
Disgust
Happiness
Sadness
Anger
Neutral
```

### AffectNet+

Para AffectNet+, foi utilizado um subconjunto balanceado com as mesmas sete classes emocionais:

```text
Surprise
Fear
Disgust
Happiness
Sadness
Anger
Neutral
```

Os datasets não estão incluídos neste repositório devido ao tamanho dos arquivos e às respectivas condições de uso.

---

## Estrutura do projeto

A estrutura esperada do repositório é:

```text
fer-vr-hmd-occlusion/
│
├── dataset/
│   ├── RAFDB/
│   └── AffectNet/
│
├── dataset_processado/
│   ├── RAFDB/
│   │   ├── Baseline/
│   │   ├── HMD/
│   │   ├── HMD_Olhos/
│   │   └── HMD_Olhos_Reais/
│   │
│   └── AffectNet/
│       ├── Baseline/
│       ├── HMD/
│       ├── HMD_Olhos/
│       └── HMD_Olhos_Reais/
│
├── metadata/
│
├── models/
│
├── POSTER_V2/
│   └── implementação e scripts de avaliação do POSTER++
│
├── QCS/
│   └── implementação e scripts de avaliação do QCS
│
├── scripts/
│   └── scripts de processamento, geração de cenários e apoio
│
├── scripts_train/
│   └── scripts relacionados a treinamento ou preparação adicional
│
├── results/
│   └── resultados, relatórios e matrizes geradas localmente
│
├── Olho_Direito.png
├── Olho_Esquerdo.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Organização dos dados processados

As imagens processadas são organizadas por dataset e por cenário.

Exemplo para AffectNet+:

```text
dataset_processado/
└── AffectNet/
    ├── Baseline/
    ├── HMD/
    ├── HMD_Olhos/
    └── HMD_Olhos_Reais/
```

A mesma lógica é aplicada ao RAF-DB.

Para manter uma comparação justa, as imagens utilizadas em cada cenário devem corresponder às mesmas amostras válidas do dataset original.

---

## Requisitos

Recomenda-se utilizar Python 3.10 ou a versão compatível com os requisitos específicos de cada modelo.

As dependências principais incluem bibliotecas para:

```text
PyTorch
OpenCV
NumPy
Pandas
MediaPipe
scikit-learn
Matplotlib
Seaborn
```

As dependências disponíveis no projeto podem ser instaladas com:

```bash
pip install -r requirements.txt
```

Como POSTER++ e QCS podem utilizar versões diferentes de algumas bibliotecas, recomenda-se manter ambientes separados para cada modelo.

---

## Criação dos ambientes virtuais

### Ambiente para POSTER++

No Windows PowerShell:

```powershell
python -m venv venv_poster
.\venv_poster\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Ambiente para QCS

No Windows PowerShell:

```powershell
python -m venv venv_qcs
.\venv_qcs\Scripts\Activate.ps1
pip install -r requirements.txt
```

Caso existam arquivos de dependências específicos dentro das pastas `POSTER_V2/` ou `QCS/`, eles devem ser utilizados de acordo com os requisitos de cada implementação.

---

## Fluxo de execução

O fluxo principal do experimento deve seguir esta ordem:

```text
1. Preparar os datasets RAF-DB e AffectNet+ localmente;
2. Configurar os caminhos dos datasets nos scripts;
3. Gerar os quatro cenários experimentais;
4. Executar a avaliação com POSTER++;
5. Executar a avaliação com QCS;
6. Gerar relatórios por classe;
7. Gerar matrizes de confusão normalizadas;
8. Comparar os resultados entre modelos, datasets e cenários.
```

---

## Geração dos cenários

Os scripts localizados em `scripts/` são responsáveis por preparar as imagens utilizadas na avaliação.

As principais etapas incluem:

```text
- carregamento da imagem original;
- detecção dos landmarks faciais;
- posicionamento da oclusão HMD;
- cálculo da rotação e dimensão da oclusão;
- inserção de olhos geométricos;
- inserção de olhos realistas;
- salvamento da imagem processada no cenário correspondente.
```

A detecção facial é utilizada para adaptar os elementos de oclusão e os overlays à posição e inclinação de cada face.

---

## Avaliação dos modelos

Após a geração das imagens processadas, os cenários são avaliados pelos modelos POSTER++ e QCS.

Para cada combinação de:

```text
modelo × dataset × cenário
```

são geradas predições para as sete classes emocionais.

Os scripts de avaliação salvam informações como:

```text
- rótulos verdadeiros;
- rótulos preditos;
- Accuracy;
- Macro F1-score;
- Balanced Accuracy;
- precisão por classe;
- recall por classe;
- F1-score por classe;
- relatórios de classificação;
- matrizes de confusão.
```

---

## Matrizes de confusão

As matrizes de confusão são utilizadas para analisar o comportamento dos modelos em cada cenário.

Nas matrizes:

```text
Linhas: rótulos verdadeiros
Colunas: rótulos preditos
Diagonal principal: porcentagem de classificações corretas
```

As matrizes permitem observar quais emoções permanecem mais robustas e quais passam a ser mais confundidas após a oclusão da região superior da face.

---

## Resultados esperados

O experimento foi projetado para comparar o desempenho dos modelos nas seguintes condições:

```text
POSTER++ × RAF-DB × 4 cenários
POSTER++ × AffectNet+ × 4 cenários
QCS × RAF-DB × 4 cenários
QCS × AffectNet+ × 4 cenários
```

Totalizando:

```text
16 configurações de avaliação
```

A comparação permite investigar:

```text
- impacto da oclusão HMD;
- efeito da inserção de olhos geométricos;
- efeito da inserção de olhos mais realistas;
- diferenças entre POSTER++ e QCS;
- diferenças entre RAF-DB e AffectNet+;
- alterações por classe emocional.
```

---

## Arquivos que não devem ser enviados ao GitHub

Os seguintes conteúdos devem permanecer locais ou ser compartilhados externamente:

```text
dataset/
dataset_processado/
results/
venv/
venv_poster/
venv_qcs/
venv_train/
*.pth
*.pt
*.ckpt
*.onnx
*.pkl
```

Esses arquivos podem ser grandes, conter pesos pré-treinados ou depender de licenças específicas.

---

## Observações importantes

- Os datasets RAF-DB e AffectNet+ não estão incluídos no repositório.
- Os pesos pré-treinados dos modelos podem precisar ser obtidos separadamente.
- Os ambientes virtuais não devem ser enviados ao GitHub.
- Resultados pesados, imagens processadas e matrizes finais podem ser armazenados em Google Drive ou outro serviço externo.
- Os arquivos `Olho_Direito.png` e `Olho_Esquerdo.png` fazem parte do pipeline de geração do cenário Real HMD-Eyes e podem ser mantidos no repositório.
- Pastas antigas ou versões não utilizadas no experimento final, como implementações legadas, devem ser removidas ou movidas para uma pasta de arquivamento antes da publicação final.

---

## Autores

- Camile Stefany da Silva
- Gustavo Camargo Domingues
- Lívia Silva Oliveira

---

## Contexto acadêmico

Este repositório foi desenvolvido para um estudo comparativo sobre reconhecimento de expressões faciais sob oclusão simulada por dispositivos de Realidade Virtual.

O foco do projeto é analisar como a cobertura da região superior da face afeta modelos de reconhecimento emocional e verificar se a inserção de informações visuais na área ocluída pode recuperar parte do desempenho perdido.

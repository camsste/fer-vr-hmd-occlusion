# RAVDESS Emotion Recognition - Frame-by-Frame Baseline

Este repositório contém um pipeline inicial para reconhecimento emocional em vídeos do dataset RAVDESS. A abordagem utilizada nesta primeira versão é frame-a-frame, ou seja, o vídeo é analisado a partir de seus frames para prever a emoção apresentada.

O objetivo é criar uma prova funcional para o projeto, permitindo comparar a emoção real do vídeo com a emoção predita pelo modelo.

## Dataset

O dataset utilizado é o **RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song**.

Nesta etapa, foram usados apenas vídeos da modalidade:

* `audio-video`
* `speech`

As emoções consideradas são:

* `neutral`
* `calm`
* `happy`
* `sad`
* `angry`
* `fearful`
* `disgust`
* `surprised`

## Ground Truth

O ground truth é extraído automaticamente a partir do nome dos arquivos do RAVDESS.

O nome dos arquivos segue o padrão:

```text
Modality-Channel-Emotion-Intensity-Statement-Repetition-Actor.mp4
```

Exemplo:

```text
01-01-03-02-01-01-01.mp4
```

Neste caso, o terceiro campo indica a emoção:

```text
03 = happy
```

Mapeamento das emoções:

```text
01 = neutral
02 = calm
03 = happy
04 = sad
05 = angry
06 = fearful
07 = disgust
08 = surprised
```

## Pipeline

O pipeline implementado realiza as seguintes etapas:

1. leitura dos vídeos do RAVDESS;
2. extração automática do ground truth;
3. seleção de 10 amostras com atores diferentes e emoções variadas;
4. reconhecimento emocional frame-a-frame;
5. geração de vídeos anotados com ground truth, predição e confiança;
6. geração de arquivos CSV com os resultados;
7. geração da matriz de confusão.

## Estrutura do projeto

```text
ravdess-emotion-demo/
│
├── dataset/
│   └── RAVDESS/
│
├── outputs/
│   ├── annotated_videos/
│   ├── figures/
│   ├── selected_10_samples.csv
│   ├── frame_results.csv
│   ├── video_summary.csv
│   ├── results_by_emotion.csv
│   ├── confusion_matrix.csv
│   └── errors.csv
│
├── scripts/
│   ├── 01_listar_amostras.py
│   ├── 02_gerar_videos_anotados.py
│   ├── 03_analisar_resultados.py
│   └── 04_gerar_matriz_confusao.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Scripts

### `01_listar_amostras.py`

Seleciona 10 vídeos do RAVDESS com atores diferentes e emoções variadas.

Gera:

```text
outputs/selected_10_samples.csv
```

### `02_gerar_videos_anotados.py`

Executa a predição emocional frame-a-frame e gera vídeos anotados.

Gera:

```text
outputs/annotated_videos/
outputs/frame_results.csv
outputs/video_summary.csv
```

### `03_analisar_resultados.py`

Calcula a acurácia, os erros e os resultados por emoção.

Gera:

```text
outputs/results_by_emotion.csv
outputs/confusion_matrix.csv
outputs/errors.csv
```

### `04_gerar_matriz_confusao.py`

Gera a imagem da matriz de confusão.

Gera:

```text
outputs/figures/confusion_matrix.png
```

## Resultados iniciais

Nesta primeira execução, foi utilizado um modelo pré-treinado de reconhecimento facial de emoções.

Foram processadas 10 amostras do RAVDESS, com atores diferentes e emoções variadas.

A acurácia inicial foi:

```text
30%
```

O modelo acertou algumas emoções, como `neutral`, `happy` e `fearful`, mas apresentou confusões frequentes entre emoções como `calm`, `sad`, `angry` e `surprised`, que foram classificadas como `neutral`.

Esse resultado indica que o pipeline está funcional, mas que o modelo pré-treinado ainda não é suficiente como solução final para o RAVDESS.

## Próximos passos

Os próximos passos planejados são:

1. adicionar oclusão na região dos olhos com uma faixa preta, simulando o uso de headset VR;
2. comparar os resultados entre vídeos originais e vídeos com oclusão;
3. implementar uma versão com olhos fake sobre a região ocluída;
4. avaliar o impacto da oclusão e dos olhos fake na classificação emocional;
5. futuramente, explorar a integração entre vídeo e áudio.

## Instalação

Crie um ambiente virtual com Python 3.11:

```bash
py -3.11 -m venv venv
```

Ative o ambiente virtual no Windows PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Dependências

O arquivo `requirements.txt` deve conter:

```text
pandas
numpy
opencv-python
tqdm
deepface
tf-keras
matplotlib
```

## Como executar

Execute os scripts nesta ordem:

```bash
cd scripts
python 01_listar_amostras.py
python 02_gerar_videos_anotados.py
python 03_analisar_resultados.py
python 04_gerar_matriz_confusao.py
```

## Observações

O dataset RAVDESS não deve ser enviado diretamente para o GitHub por causa do tamanho dos arquivos.

A pasta `dataset/` deve ser mantida localmente.

Os vídeos anotados e os resultados mais pesados podem ser compartilhados externamente, por exemplo, via Google Drive.

## Status atual

Pipeline funcional implementado.

* 10 amostras processadas
* atores diferentes
* ground truth extraído automaticamente
* vídeos anotados gerados
* arquivos CSV gerados
* matriz de confusão gerada
* acurácia inicial: 30%

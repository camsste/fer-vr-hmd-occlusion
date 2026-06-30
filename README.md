# Facial Expression Recognition under Simulated VR HMD Occlusion

This repository contains the code developed to evaluate the impact of Virtual Reality Head-Mounted Display (HMD) occlusion on Facial Expression Recognition (FER).

The experiment compares two FER models, **POSTER++** and **QCS**, using the **RAF-DB** and **AffectNet+** datasets under four facial-visibility scenarios:

- Baseline
- HMD
- HMD-Eyes
- Real HMD-Eyes

The goal is to investigate how upper-face occlusion, especially around the eyes and eyebrows, affects model performance and whether inserting visual eye information can partially mitigate this degradation.

---

## Overview

Facial Expression Recognition models usually rely on information from the entire face, including the eyes, eyebrows, forehead, mouth, and spatial relationships between facial regions.

In Virtual Reality applications, however, HMD devices cover a substantial part of the upper face. To study this effect in a controlled way, this project generates different versions of the same facial images and evaluates the models under equivalent experimental conditions.

The implemented pipeline performs the following steps:

1. Organizes the original images from RAF-DB and AffectNet+;
2. Detects facial landmarks;
3. Generates simulated HMD occlusion;
4. Inserts geometric and realistic eye representations;
5. Evaluates the images using POSTER++;
6. Evaluates the images using QCS;
7. Computes global and class-level metrics;
8. Generates classification reports and normalized confusion matrices.

---

## Experimental Scenarios

Four versions are generated for each image.

### 1. Baseline

Original facial image without any modification.

```text
Baseline
```

This condition is used as the reference scenario, where the complete face is visible.

---

### 2. HMD

A synthetic occlusion is applied to the upper facial region to simulate a Virtual Reality headset.

```text
HMD
```

The occlusion covers mainly the forehead, eyebrows, and part of the eye region.

---

### 3. HMD-Eyes

The HMD occlusion remains in place, but geometric eye representations are inserted into the covered region.

```text
HMD_Olhos
```

This scenario uses simple visual elements to represent the eyes, including geometric shapes for the sclera and iris.

---

### 4. Real HMD-Eyes

The HMD occlusion remains in place, but more realistic eye textures are inserted into the occluded region.

```text
HMD_Olhos_Reais
```

The eye textures are resized, positioned, and rotated according to facial landmarks.

The overlay files used in this scenario are:

```text
Olho_Direito.png
Olho_Esquerdo.png
```

---

## Evaluated Models

Two Facial Expression Recognition models are evaluated under the same datasets, scenarios, and metrics.

### POSTER++

POSTER++ is a transformer-based FER model that combines facial information through feature fusion mechanisms.

Its implementation and evaluation scripts are located in:

```text
POSTER_V2/
```

### QCS

QCS, or Quadruplet Cross Similarity, is a FER model based on feature refinement through cross-similarity learning.

Its implementation and evaluation scripts are located in:

```text
QCS/
```

The models are executed independently because they have different dependencies, pretrained weights, and configurations.

---

## Datasets

The experiments use two in-the-wild facial expression recognition datasets.

### RAF-DB

RAF-DB is evaluated using seven emotion classes:

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

A balanced AffectNet+ subset is evaluated using the same seven emotion classes:

```text
Surprise
Fear
Disgust
Happiness
Sadness
Anger
Neutral
```

The datasets are not included in this repository due to size limitations and dataset usage restrictions.

---

## Repository Structure

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
├── POSTER_V2/
│   ├── 08_avaliar_poster.py
│   ├── 08.1_avaliar_poster_affectnet.py
│   └── requirements.txt
│
├── QCS/
│   ├── 09_avaliar_qcs_rafdb.py
│   ├── 10_avaliar_qcs_affectnet.py
│   └── requirements.txt
│
├── scripts/
│   ├── 03_gerador_hmd.py
│   ├── 04_injetor_olhos.py
│   ├── 05_pipeline_geral.py
│   ├── 06_pipeline_rafdb.py
│   ├── 06.2_pipeline_rafdb_olhos_reais.py
│   ├── 07_pipeline_affectnet.py
│   └── 11_pipeline_affectnet_passagem_unica.py
│
├── Olho_Direito.png
├── Olho_Esquerdo.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Processed Dataset Organization

The processed images are organized by dataset and scenario.

Example for AffectNet+:

```text
dataset_processado/
└── AffectNet/
    ├── Baseline/
    ├── HMD/
    ├── HMD_Olhos/
    └── HMD_Olhos_Reais/
```

The same structure is used for RAF-DB.

To ensure a fair comparison, the samples evaluated across all scenarios must correspond to the same valid images from the original dataset.

---

## Requirements

The root `requirements.txt` contains the dependencies required for preprocessing and scenario generation.

```bash
pip install -r requirements.txt
```

POSTER++ and QCS have their own dependency files:

```text
POSTER_V2/requirements.txt
QCS/requirements.txt
```

Separate virtual environments are recommended because the models may require different PyTorch and library versions.

---

## Environment Setup

### Scenario Generation Environment

```powershell
python -m venv venv_gerador
.\venv_gerador\Scripts\Activate.ps1
pip install -r requirements.txt
```

### POSTER++ Environment

```powershell
python -m venv venv_poster
.\venv_poster\Scripts\Activate.ps1
pip install -r .\POSTER_V2\requirements.txt
```

### QCS Environment

```powershell
python -m venv venv_qcs
.\venv_qcs\Scripts\Activate.ps1
pip install -r .\QCS\requirements.txt
```

---

## Workflow

The main experimental workflow is:

```text
1. Obtain RAF-DB and AffectNet+ locally;
2. Configure the dataset paths in the scripts;
3. Generate the four experimental scenarios;
4. Run POSTER++ evaluation;
5. Run QCS evaluation;
6. Save predictions and metrics;
7. Generate classification reports;
8. Generate normalized confusion matrices;
9. Compare performance across models, datasets, and scenarios.
```

---

## Scenario Generation

The scripts in the `scripts/` directory are responsible for generating the processed images.

The main steps include:

```text
- loading the original image;
- detecting facial landmarks;
- positioning the HMD occlusion;
- computing occlusion rotation and dimensions;
- inserting geometric eyes;
- inserting realistic eye textures;
- saving the processed image in the corresponding scenario folder.
```

Facial landmarks are used to adapt the occlusion and eye overlays to the position, scale, and inclination of each face.

---

## Model Evaluation

After generating the processed datasets, the images are evaluated using POSTER++ and QCS.

Each evaluation corresponds to a combination of:

```text
Model × Dataset × Scenario
```

The scripts generate predictions for the seven emotion classes and save metrics such as:

```text
- Accuracy;
- Macro F1-score;
- Balanced Accuracy;
- Precision per class;
- Recall per class;
- F1-score per class;
- Classification reports;
- Confusion matrices.
```

---

## Confusion Matrices

Normalized confusion matrices are used to analyze class-level changes under occlusion.

```text
Rows: true labels
Columns: predicted labels
Main diagonal: percentage of correctly classified samples
```

These matrices help identify which emotions remain comparatively robust and which emotions become more frequently confused after upper-face occlusion.

---

## Evaluation Scope

The experiment evaluates:

```text
POSTER++ × RAF-DB × 4 scenarios
POSTER++ × AffectNet+ × 4 scenarios
QCS × RAF-DB × 4 scenarios
QCS × AffectNet+ × 4 scenarios
```

This results in:

```text
16 evaluation configurations
```

The comparison investigates:

```text
- the impact of HMD occlusion;
- the effect of geometric eye insertion;
- the effect of realistic eye insertion;
- differences between POSTER++ and QCS;
- differences between RAF-DB and AffectNet+;
- class-level changes under occlusion.
```

---

## Files Not Included in the Repository

The following files and directories remain local or should be shared through external storage:

```text
dataset/
dataset_processado/
venv/
venv_gerador/
venv_poster/
venv_qcs/
venv_train/
*.pth
*.pt
*.ckpt
*.onnx
*.pkl
```

These files may be large, contain pretrained weights, or be subject to dataset and model usage restrictions.

---

## Notes

- RAF-DB and AffectNet+ are not included in this repository.
- Pretrained model weights may need to be obtained separately.
- Virtual environments should not be uploaded to GitHub.
- Large processed datasets and result files can be stored externally, such as in Google Drive.
- `Olho_Direito.png` and `Olho_Esquerdo.png` are included because they are part of the Real HMD-Eyes generation pipeline.
- The POSTER++ and QCS source code is maintained with the corresponding license files provided in their directories.

---

## Authors

- Camile Stefany da Silva
- Gustavo Camargo Domingues
- Lívia Silva Oliveira

---

## Academic Context

This repository was developed for a comparative study on Facial Expression Recognition under simulated Virtual Reality HMD occlusion.

The study investigates how upper-face occlusion affects current FER models and whether generic eye reconstruction strategies can recover part of the facial information lost by the headset.

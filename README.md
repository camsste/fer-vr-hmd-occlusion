# Facial Expression Recognition under Simulated VR Occlusion

This repository contains the implementation developed for a comparative study of Facial Expression Recognition (FER) under simulated Virtual Reality Head-Mounted Display (HMD) occlusion.

The project evaluates how occluding the upper facial region affects emotion recognition models and investigates whether inserting visual eye information can reduce the performance degradation caused by the headset.

The implementation compares two FER architectures, POSTER++ and QCS, using the RAF-DB and AffectNet+ datasets.

---

## Overview

Facial expression recognition models usually rely on visual information from the complete face, including the eyes, eyebrows, forehead, nose, and mouth.

However, in Virtual Reality applications, Head-Mounted Displays cover a substantial part of the upper face. This repository implements a controlled evaluation pipeline to analyze how this occlusion affects facial emotion recognition.

For each image, the pipeline generates three visual conditions:

1. **Baseline**  
   Original facial image without modifications.

2. **HMD**  
   Upper-face occlusion simulating a Virtual Reality headset.

3. **HMD-Eyes**  
   HMD occlusion with eye information inserted into the occluded region.

The same scenarios are evaluated using both POSTER++ and QCS.

---

## Evaluated Models

- **POSTER++**
- **QCS**

The models are evaluated independently under the same datasets, image subsets, experimental conditions, and metrics.

---

## Datasets

The experiments use two facial expression recognition datasets:

- **RAF-DB**
- **AffectNet+**

Both datasets are evaluated using the following emotion classes:

```text
Surprise
Fear
Disgust
Happiness
Sadness
Anger
Neutral

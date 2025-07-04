# COMSYS Hackathon 2025 – Task A: Gender Classification

This repository contains the PyTorch implementation for **Task A** of COMSYS Hackathon-5, 2025. The goal is to classify gender (Male/Female) from facial images captured under **adverse visual conditions** such as fog, blur, rain, and poor lighting.

---

## 📁 Dataset Structure

The dataset used is **FACECOM**, organized as follows:

Task_A/
├── train/
│ ├── male/
│ └── female/
├── val/
│ ├── male/
│ └── female/
└── test/
├── male/
└── female/


Each folder contains facial images under degraded environmental conditions.

---

## 🧠 Model Overview

- **Architecture:** Pretrained **ResNet18** with modified final layer
- **Framework:** PyTorch
- **Input Size:** 224 × 224 RGB images
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (learning rate = 1e-4)
- **Metrics:** Accuracy, Precision, Recall, F1-score

---
## Evaluation metrices:
Classification Report:

              precision    recall  f1-score   support

      female       0.77      0.85      0.81        79
        male       0.96      0.94      0.95       343

    accuracy                           0.92       422
   macro avg       0.87      0.89      0.88       422
weighted avg       0.93      0.92      0.93       422
## 🔧 Environment Setup

Install the required dependencies:

```bash
pip install -r requirements.txt


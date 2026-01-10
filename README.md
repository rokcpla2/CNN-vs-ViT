# Comparative Analysis of Inductive Bias and Data Efficiency: CNN vs. Vision Transformer

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS-000000?style=flat-square&logo=apple&logoColor=white)]()

## 📝 Abstract
This study investigates the impact of **Data Scale** on the performance of **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)**. We empirically verify the hypothesis that ViTs lack the inductive bias inherent in CNNs, making them more "data-hungry" and difficult to generalize in low-data regimes.

By scaling the CIFAR-10 training dataset from **10% to 100%**, we analyze the convergence speed, generalization capability, and overfitting tendencies of both architectures. Additionally, this project proposes a **Hybrid Computing Strategy**, utilizing **Apple Silicon (MPS)** for efficient edge-based CNN training and **Cloud TPU (v2/v5e)** for high-throughput ViT training.

---

## 1. Introduction

### 1.1. Motivation
In the field of Computer Vision, the paradigm is shifting from CNNs to Transformers. However, deploying ViTs in real-world scenarios typically requires massive datasets (e.g., JFT-300M, ImageNet). This project aims to quantify the **performance gap** between a standard CNN (ResNet) and a lightweight ViT under restricted data environments, providing insights for efficient model selection in data-scarce applications.

### 1.2. Key Contributions
* **Data Sensitivity Analysis:** Quantitative comparison of model performance across four data scales (10%, 25%, 50%, 100%).
* **Hybrid Training Pipeline:** Demonstration of a heterogeneous computing environment optimizing resource allocation (Edge Device vs. Cloud Accelerator).
* **Verification of Inductive Bias:** Empirical evidence supporting the necessity of locality and translation invariance in low-data regimes.

---

## 2. Experimental Methodology

### 2.1. Model Architectures
We compared two representative models with similar parameter scales:
* **CNN:** `ResNet-18` (He et al.)
    * *Characteristics:* Strong inductive bias (Locality, Translation Invariance).
* **ViT:** `vit_tiny_patch16_224` (Dosovitskiy et al.)
    * *Characteristics:* Long-range dependency modeling via Self-Attention, low inductive bias.

### 2.2. Controlled Variables (Hyperparameters)
To ensure a fair comparison, all environmental variables were strictly controlled:
* **Dataset:** CIFAR-10 (Resized to 224x224 for ViT compatibility)
* **Data Scales:** `[10%, 25%, 50%, 100%]` of the Training Set
* **Epochs:** 50 (Fixed)
* **Batch Size:** 128
* **Optimizer:** AdamW (`lr=1e-3`, `weight_decay=1e-4`)
* **Seed:** `42` (For reproducibility)

### 2.3. Computing Environment (Hybrid Strategy)
| Architecture | Hardware | Framework | Rationale |
| :--- | :--- | :--- | :--- |
| **CNN** | **MacBook Air M3** (16GB) | PyTorch (MPS) | High efficiency for sequential operations; suitable for ResNet. |
| **ViT** | **Google Colab TPU** (v2/v5e) | PyTorch XLA | Massive parallel processing required for Multi-Head Attention. |

---

## 3. Results & Analysis 📊

### 3.1. Overall Performance Summary
The table below summarizes the Test Accuracy changes as the data scale increases.

| Data Ratio (Images) | CNN (ResNet-18) | ViT (Tiny) | **Performance Gap** | Improvement (ViT) |
| :---: | :---: | :---: | :---: | :---: |
| **10%** (5k) | 63.40% | 45.01% | **+18.39%** | - |
| **25%** (12.5k) | 72.01% | 55.30% | +16.71% | +10.29% |
| **50%** (25k) | 79.13% | 65.24% | +13.89% | +9.94% |
| **100%** (50k) | **82.23%** | **73.33%** | **+8.90%** | +8.09% |

### 3.2. Key Findings

#### 📉 1. The Gap is Closing (Scalability of ViT)
* At **10% data**, the gap was substantial (**18.39%**), confirming that ViT struggles to learn visual representations without sufficient examples.
* As data increased to **100%**, the gap narrowed significantly to **8.90%**.
* **Insight:** ViT exhibits a steeper learning curve than CNN. This suggests that with even more data (e.g., ImageNet), ViT has the potential to outperform CNNs, adhering to the *scaling laws* of Transformers.

#### 🧠 2. Inductive Bias vs. Data Scale
* **CNN** showed stable performance even with 10% data (63.4%), benefiting from its architectural priors (Convolution).
* **ViT** required at least 50% data (25k images) to reach a respectable accuracy (>60%), proving its "data-hungry" nature.

#### ⚠️ 3. Overfitting in ViT
* At 100% data, ViT achieved a **Training Accuracy of 95.81%** but a **Test Accuracy of 73.33%**.
* **Analysis:** The model has sufficient capacity to memorize the training data but fails to generalize perfectly. This indicates the need for strong regularization techniques (e.g., Mixup, CutMix, Augmentation) when training ViTs on small datasets like CIFAR-10.

---

## 4. Conclusion & Future Work

### Conclusion
This study empirically validates that **CNNs are more data-efficient** and suitable for scenarios with limited data resources. Conversely, **ViTs show higher scalability** but require significantly more data or strong regularization to overcome the lack of inductive bias. For practical applications with <50k images, ResNet remains the superior choice unless pre-training is applied.

### Future Work
1.  **Strong Augmentation:** Apply AutoAugment, Mixup, or CutMix to mitigate ViT's overfitting.
2.  **Edge-Cloud Collaboration:** Develop an adaptive inference system that runs lightweight CNNs on edge devices and offloads difficult samples to a cloud-based ViT (e.g., `ViT-Large`), utilizing the hybrid environment established in this project.
3.  **Pre-training:** Compare results when initializing ViT with weights pre-trained on ImageNet-1k.

---

## 5. Directory Structure

```bash
CNN-vs-ViT/
├── data/                   # CIFAR-10 Dataset
├── models/                 # Model Definitions
├── train_cnn.py            # Optimized for Mac M3 (MPS)
├── train_vit_tpu.py        # Optimized for Colab TPU (XLA)
├── utils.py                # Data Loaders & Plotting Tools
├── README.md               # Project Report
└── results/                # Logs & Saved Plots

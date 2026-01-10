# Comparative Analysis of Inductive Bias and Data Efficiency: CNN vs. Vision Transformer

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS-000000?style=flat-square&logo=apple&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📝 Abstract

This study investigates the impact of **Data Scale** on the performance of **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)**. We empirically verify the hypothesis that ViTs lack the inductive bias inherent in CNNs, making them more "data-hungry" and difficult to generalize in low-data regimes.

By scaling the CIFAR-10 training dataset from **10% to 100%**, we analyze the convergence speed, generalization capability, and overfitting tendencies of both architectures. Additionally, this project demonstrates a **Hybrid Computing Strategy**, utilizing **Apple Silicon (MPS)** for efficient edge-based CNN training and **Cloud TPU (XLA)** for high-throughput ViT training.

---

## 1. Introduction

### 1.1 Motivation

In the field of Computer Vision, the paradigm is shifting from CNNs to Transformers. However, deploying ViTs in real-world scenarios typically requires massive datasets (e.g., JFT-300M, ImageNet). This project aims to quantify the **performance gap** between a standard CNN (ResNet) and a lightweight ViT under restricted data environments, providing insights for efficient model selection in data-scarce applications.

### 1.2 Key Contributions

- **Data Sensitivity Analysis:** Quantitative comparison of model performance across four data scales (10%, 25%, 50%, 100%).
- **Hybrid Training Pipeline:** Heterogeneous computing environment optimizing resource allocation (Edge Device vs. Cloud Accelerator).
- **Verification of Inductive Bias:** Empirical evidence supporting the necessity of locality and translation invariance in low-data regimes.

---

## 2. Experimental Methodology

### 2.1 Model Architectures

We compared two representative models with similar parameter scales:

#### CNN
- **Model:** ResNet-18 (He et al.)
- **Characteristics:** Strong inductive bias (Locality, Translation Invariance)
- **Optimization:** Modified first conv layer (kernel 3×3) for CIFAR-10 (32×32)

#### Vision Transformer
- **Model:** vit_tiny_patch16_224 (Dosovitskiy et al.)
- **Characteristics:** Long-range dependency modeling via Self-Attention, low inductive bias
- **Preprocessing:** Resize images to 224×224

### 2.2 Computing Environment (Hybrid Strategy)

| Architecture | Hardware | Framework | Script |
|:-------------|:---------|:----------|:-------|
| **CNN** | **MacBook Air M3** (16GB) | PyTorch (MPS) | `train_cnn.py` |
| **ViT** | **Google Colab TPU** (v2) | PyTorch XLA | `train_vit.py` |

---

## 3. Directory Structure

```bash
CNN-vs-ViT/
├── train_cnn.py            # Script for Mac M3 (MPS)
├── train_vit.py            # Script for Cloud TPU (XLA)
├── requirements.txt        # Dependencies
└── README.md               # Project Report
```

---

## 4. Usage

### 4.1 Installation

```bash
git clone https://github.com/YOUR_USERNAME/CNN-vs-ViT.git
cd CNN-vs-ViT
pip install -r requirements.txt
```

### 4.2 Train CNN (Local Edge Device)

Optimized for Apple Silicon (MPS). You can control the data ratio using the `--ratio` argument.

```bash
# Train with 10% data (Fast experiment)
python train_cnn.py --ratio 0.1 --epochs 50

# Train with 100% data (Full experiment)
python train_cnn.py --ratio 1.0 --epochs 50
```

### 4.3 Train ViT (Cloud TPU)

Optimized for TPU environments (e.g., Google Colab).

```bash
# Train with 25% data
python train_vit.py --ratio 0.25 --epochs 50
```

---

## 5. Experimental Results 📊

### 5.1 Performance Comparison Graph

The graph below illustrates the Test Accuracy trends as the dataset size increases.

### 5.2 Numerical Analysis

| Data Ratio | CNN (ResNet-18) | ViT (Tiny) | Performance Gap |
|:-----------|:----------------|:-----------|:----------------|
| 10% (5k) | 63.40% | 45.01% | +18.39% |
| 25% (12.5k) | 72.01% | 55.30% | +16.71% |
| 50% (25k) | 79.13% | 65.24% | +13.89% |
| 100% (50k) | 82.23% | 73.33% | +8.90% |

### 5.3 Key Findings

#### Inductive Bias Matters in Low Data
CNN consistently outperformed ViT, especially when data was scarce (10%), thanks to its architectural priors.

#### ViT is "Data Hungry"
ViT showed a steeper learning curve. The performance gap closed from 18.4% (at 10% data) to 8.9% (at 100% data).

#### Overfitting
At 100% data, ViT achieved 95.8% Training Acc but only 73.3% Test Acc, indicating a need for stronger regularization (e.g., Mixup, CutMix) or more data.

---

## 6. Conclusion

This study empirically validates that CNNs are more data-efficient and suitable for edge-based scenarios with limited resources. Conversely, ViTs show higher scalability but require significantly more data to overcome the lack of inductive bias.

### Future Work

We plan to develop an **Adaptive Edge-Cloud Inference System** that dynamically offloads difficult samples to a cloud-based ViT while processing easy samples on a local CNN, leveraging the hybrid environment established in this project.

---

## 👨‍💻 Author

**Minkyu Kim**  
Dept. of Electronic Engineering, KNUT  
Research Interest: Embedded AI, FPGA Acceleration, Computer Vision

---

## 📄 License

This project is licensed under the MIT License.

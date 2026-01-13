# CNN vs ViT: 데이터 규모에 따른 성능 비교 연구
Data-Scale Sensitivity Analysis of CNN (ResNet-18) and ViT (Tiny-Patch16)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS-000000?style=flat-square&logo=apple&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📝 연구 개요

이 연구는 데이터 규모(Data Scale)가 CNN(ResNet-18)과 Vision Transformer(ViT-Tiny)의 모델 성능에 어떤 영향을 미치는지 비교·분석하기 위해 수행되었습니다.

특히, CNN이 가진 inductive bias와 ViT의 데이터 의존성이 실제 실험에서 어떤 형태로 나타나는지 검증하는 것을 목표로 합니다.

---

## 1. 연구 배경

### 1.1 CNN 
- Convolution(필터)을 사용해 지역적 특징을 탐색
- 이미지 구조에 특화된 inductive bias 내장
- 적은 데이터에서도 안정적으로 성능 확보
  
### 1.2 Vision Transformer
- 이미지를 패치(patch) 단위로 분할해 토큰처럼 처리
- Self-Attention으로 전역 관계를 학습
- 이미지 구조에 대한 선천적 가정이 거의 없음 → 많은 데이터 필요

### 1.3 연구 질문
- 데이터 비율(10%, 25%, 50%, 100%) 변화가 두 모델의 성능에 어떤 차이를 만드는가?
- 작은 데이터 상황에서 CNN이 더 강력한 이유는 무엇인가?
- ViT는 어느 시점에서 CNN과 성능 격차가 줄어드는가?

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
├── README.md               # Project Report 
├── requirements.txt        # Dependencies
├── train_cnn.py            # Script for Mac M3 (MPS)
└── train_vit.py            # Script for Cloud TPU (XLA)
```

---

## 4. Usage

### 4.1 Installation

```bash
git clone https://github.com/rokcpla2/CNN-vs-ViT.git
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

<p align="center">
  <img src="https://github.com/user-attachments/assets/18137eea-70eb-4908-92b0-5c636110ddbb" width="845" height="573" alt="final_result_dark">
</p>

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

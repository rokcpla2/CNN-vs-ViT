# CNN vs ViT: 데이터 규모에 따른 성능 비교 연구
### Data-Scale Sensitivity Analysis of CNN (ResNet-18) and ViT (Tiny-Patch16)

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

## 2. 실험 환경

### 2.1 데이터셋
- CIFAR-10 (32×32 RGB, 10 classes)
- 사용 비율: 10%, 25%, 50%, 100%

### 2.2 모델
CNN: ResNet-18
- CIFAR-10에 맞게 3×3 Conv로 수정

ViT: ViT-Tiny-Patch16
224×224 Resize 후 Patch-Embedding
Self-Attention 기반 구조


### 2.3 컴퓨팅 환경

| Architecture | Hardware | Framework | Script |
|:-------------|:---------|:----------|:-------|
| CNN | MacBook Air M3 (16GB) | PyTorch | `train_cnn.py` |
| ViT | Google Colab TPU | PyTorch XLA | `train_vit.py` |

---

## 3. 코드 구조

```bash
CNN-vs-ViT/
├── README.md               # Project Report 
├── requirements.txt        # Dependencies
├── train_cnn.py            # Script for Mac M3 (MPS)
└── train_vit.py            # Script for Cloud TPU (XLA)
```

---

## 4. 실행 방법

### 4.1 설치

```bash
git clone https://github.com/rokcpla2/CNN-vs-ViT.git
cd CNN-vs-ViT pip install -r requirements.txt
```

### 4.2 CNN 학습 (Local Edge Device)

Optimized for Apple Silicon (MPS). You can control the data ratio using the `--ratio` argument.

```bash
# Train with 10% data (Fast experiment)
python train_cnn.py --ratio 0.1 --epochs 50

# Train with 100% data (Full experiment)
python train_cnn.py --ratio 1.0 --epochs 50
```

### 4.3 ViT 학습 (Cloud TPU)

Optimized for TPU environments (e.g., Google Colab).

```bash
# Train with 25% data
python train_vit.py --ratio 0.25 --epochs 50
```

---

## 5. 실험 결과 📊

### 5.1 Accuracy 비교 그래프

<p align="center">
  <img src="https://github.com/user-attachments/assets/18137eea-70eb-4908-92b0-5c636110ddbb" width="845" height="573" alt="final_result_dark">
</p>

### 수치 비교

| Data Ratio | CNN | ViT | Performance Gap |
|:-----------|:----------------|:-----------|:----------------|
| 10% (5k) | 63.40% | 45.01% | +18.39% |
| 25% (12.5k) | 72.01% | 55.30% | +16.71% |
| 50% (25k) | 79.13% | 65.24% | +13.89% |
| 100% (50k) | 82.23% | 73.33% | +8.90% |

## 6. 분석 (Analysis)

### 6.1 CNN은 왜 데이터가 적어도 강한가?
- 지역적 패턴을 우선적으로 보는 inductive bias
- 필터가 전체 이미지에 공유됨
- 학습해야 할 파라미터 공간이 상대적으로 좁음

### 6.2 ViT는 왜 많은 데이터가 필요한가?
- 패치 간 관계를 전부 학습해야 함
- 이미지 구조에 대한 사전 가정이 없음
- 작은 데이터에서는 쉽게 overfitting 발생

### 6.3 성능 격차가 줄어드는 지점
- 데이터가 많아질수록 ViT의 장점(전역적 특징 학습)이 발휘됨
- 100% 데이터 구간에서는 Gap이 약 8.9%까지 감소

---

## 6. 결론(Conclusion)

- 작은 데이터에서는 CNN이 압도적으로 유리
- ViT는 대규모 데이터에서 점차 CNN과 격차를 좁힘
- 모델 선택은 데이터 조건에 따라 달라져야 함

### 향후 연구 방향 (Future Work)

- Mixup/CutMix 기반 ViT regularization 실험
- augmentation 영향 비교
- patch size 변화(ablation study)
- CNN-ViT 협력 구조(Edge-Cloud Hybrid Inference) 연구

---

## 👨‍💻 Author

**Minkyu Kim**  
Dept. of Electronic Engineering, KNUT  
Research Interest: Embedded AI, FPGA Acceleration, Computer Vision

---

## 📄 License

This project is licensed under the MIT License.

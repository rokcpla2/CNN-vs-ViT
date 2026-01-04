# CNN-vs-ViT: Data Scale Sensitivity Study

## 1. Introduction
This project investigates the performance gap between **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** under varying data regimes. 

We hypothesize that **"ViTs are more data-hungry than CNNs"** and aim to observe how the performance gap narrows or widens as the dataset size increases on CIFAR-10. 

Additionally, this study adopts a **Hybrid Environment Strategy**, demonstrating how to optimize training pipelines for different hardware architectures: **Edge Devices (Mac M3/MPS) for efficient CNN training** vs. **Cloud Accelerators (Colab TPU v5e-1) for high-throughput ViT training**.

## 2. Methodology

### Models
* **CNN:** ResNet-18 (He et al.) - Efficient inductive bias for small data.
* **ViT:** Vision Transformer (`vit_tiny_patch16_224`) - Leveraging `timm` library.

### Experimental Design (Controlled Variables)
To ensure a fair comparison, the following hyperparameters were strictly controlled across both environments:

* **Dataset:** CIFAR-10
* **Data Scales:** `[10%, 25%, 50%, 100%]` of Training Data
* **Epochs:** 50 (fixed for all experiments)
* **Batch Size:** 128 (Unified for stability)
* **Optimizer:** AdamW (`lr=0.001`, `weight_decay=1e-4`)
* **Seed:** Fixed to `42` for reproducibility

### Hardware-Specific Optimization
* **CNN (Mac M3):** Optimized for **MPS (Metal Performance Shaders)** with `num_workers=0` to eliminate multiprocessing overhead on MacOS.
* **ViT (Colab TPU):** Optimized for **TPU v5e-1 (Tensor Processing Unit)** using `torch_xla` and `MpDeviceLoader` for massive parallel processing.

## 3. Tech Stack
* **Framework:** PyTorch
* **Library:** `timm` (PyTorch Image Models), `torchvision`, `torch_xla`
* **Environments (Hybrid Strategy):**
    *  **Local:** MacBook Air M3 16GB (Apple Silicon MPS) - *Used for CNN*
    *  **Cloud:** Google Colab (TPU v5e-1) - *Used for ViT*

## 4. Usage
### Train CNN (Local Mac M3)
```bash
python train_cnn.py --epochs 50 --batch_size 128 --ratio 0.1
```

## 5. Experiment Results 📊

We observed the performance changes as the data scale increased from **10% to 25%**.

### 5.1. Data Scale: 10% (5,000 Images)
* **Goal:** Test performance in an extremely low-data regime.

| Model | Platform | Test Accuracy | Training Time |
| :--- | :--- | :--- | :--- |
| **CNN (ResNet-18)** | Mac M3 (MPS) | **63.40%** | ~3.5 min |
| **ViT (Tiny-Patch16)** | Colab (TPU v5e-1) | 45.01% | ~11.4 min |

> **Analysis:** > CNN outperformed ViT by **+18.39%**. ViT failed to generalize, showing severe overfitting (Train 75% vs Test 45%).

<br>

### 5.2. Data Scale: 25% (12,500 Images)
* **Goal:** Observe if ViT starts to catch up with 2.5x more data.

| Model | Platform | Test Accuracy | Improvement (vs 10%) |
| :--- | :--- | :--- | :--- |
| **CNN (ResNet-18)** | Mac M3 (MPS) | **72.01%** | +8.61% |
| **ViT (Tiny-Patch16)** | Colab (TPU v5e-1) | 55.30% | +10.29% |

> **Analysis:** > Both models improved significantly, but the **performance gap remains large (~16.7%)**. 
> ViT's accuracy jumped by over 10%, yet it still lags behind CNN, confirming its high data dependency.
>
> <br>

### 5.3. Data Scale: 50% (25,000 Images)
* **Goal:** Verify if the gap narrows as ViT gets more data.

| Model | Platform | Test Accuracy | Improvement (vs 25%) |
| :--- | :--- | :--- | :--- |
| **CNN (ResNet-18)** | Mac M3 (MPS) | **79.13%** | +7.12% |
| **ViT (Tiny-Patch16)** | Colab (TPU v5e-1) | 65.24% | +9.94% |

> **Analysis:** > **The gap is closing!** (16.7% → 13.9%). 
> ViT finally surpassed 60% accuracy, showing a steeper learning curve than CNN. 
> However, severe overfitting persists (Train 94% vs Test 65%), indicating that 25k images are still insufficient for ViT to fully generalize without strong regularization.

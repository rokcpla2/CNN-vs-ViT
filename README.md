# CNN-vs-ViT: Data Scale Sensitivity Study

## 1. Introduction
This project investigates the performance gap between **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** under varying data regimes. 

We hypothesize that **"ViTs are more data-hungry than CNNs"** and aim to observe how the performance gap narrows or widens as the dataset size increases on CIFAR-10. 

Additionally, this study adopts a **Hybrid Environment Strategy**, demonstrating how to optimize training pipelines for different hardware architectures: **Edge Devices (Mac M3/MPS) for efficient CNN training** vs. **Cloud GPUs (Colab T4/CUDA) for compute-intensive ViT training**.

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
* **ViT (Colab T4):** Optimized for **CUDA** with `num_workers=2` to utilize parallel data prefetching (Resize operations) on Linux.

## 3. Tech Stack
* **Framework:** PyTorch
* **Library:** `timm` (PyTorch Image Models), `torchvision`
* **Environments (Hybrid Strategy):**
    * 💻 **Local:** MacBook Air M3 (Apple Silicon MPS) - *Used for CNN*
    * ☁️ **Cloud:** Google Colab (NVIDIA T4 GPU) - *Used for ViT*

## 4. Usage
### Train CNN (Local Mac M3)
```bash
python train_cnn.py --epochs 50 --batch_size 128 --ratio 0.1


## 5. Preliminary Results (Data Scale: 10%)

We conducted the first experiment using only **10% (5,000 images)** of the CIFAR-10 training set to test the data efficiency of each model.

| Model | Platform | Test Accuracy | Training Time |
| :--- | :--- | :--- | :--- |
| **CNN (ResNet-18)** | Mac M3 (MPS) | **63.40%** | ~3.5 min |
| **ViT (Tiny-Patch16)** | Colab (T4 GPU) | 45.96% | ~30.0 min |

### 🔍 Analysis
* **Performance Gap:** CNN outperformed ViT by **+17.44%** in the low-data regime.
* **Interpretation:** This strongly supports the hypothesis that ViTs lack the *inductive bias* (locality, translation invariance) inherent in CNNs, making them difficult to train effectively with limited data (5k samples).
* **Next Steps:** We will extend this experiment to **25%, 50%, and 100%** data scales to observe if ViT catches up with CNNs as data volume increases.

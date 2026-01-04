# CNN-vs-ViT
I’m curious about how data scale affects the performance gap between CNNs and Vision Transformers.

# CNN-vs-ViT: Data Scale Sensitivity Study

## 1. Introduction
This project investigates the performance gap between **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** under varying data regimes.
Specifically, I aim to verify the hypothesis that **ViTs are more data-hungry than CNNs** and observe how the performance gap narrows as the dataset size increases on CIFAR-10.

## 2. Methodology
### Models
- **CNN**: ResNet-18 (He et al.)
- **ViT**: Vision Transformer (Dosovitskiy et al.) - *Using `timm` implementation*

### Experimental Design (Controlled Variables)
- **Dataset**: CIFAR-10
- **Data Scales**: [10%, 25%, 50%, 100%] of Training Data
- **Epochs**: 50 (fixed for all experiments)
- **Optimizer**: AdamW
- **Augmentation**: RandomCrop + HorizontalFlip (Applied consistently)

## 3. Tech Stack
- **Framework**: PyTorch
- **Library**: `timm` (PyTorch Image Models)
- **Environment**: Google Colab (T4 GPU)

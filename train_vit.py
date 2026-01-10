import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import os
import numpy as np
import timm

# TPU / XLA Libraries
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from torch.utils.data import DataLoader, Subset

def get_args():
    parser = argparse.ArgumentParser(description='Train ViT (Tiny) on CIFAR-10 with TPU Acceleration')
    parser.add_argument('--ratio', type=float, default=1.0, help='Dataset ratio to use (0.1 to 1.0)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    return parser.parse_args()

def set_seed(seed):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_data_loaders(ratio, batch_size, num_workers):
    """
    Prepares CIFAR-10 data loaders for ViT.
    Resizes images to 224x224 to fit standard Vision Transformer input requirements.
    """
    # ViT requires 224x224 input resolution
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download dataset
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Subset creation for data efficiency experiments
    if ratio < 1.0:
        num_train = len(full_trainset)
        indices = list(range(num_train))
        split = int(np.floor(ratio * num_train))
        
        # Consistent shuffling with fixed seed
        np.random.shuffle(indices)
        train_subset = Subset(full_trainset, indices[:split])
    else:
        train_subset = full_trainset

    # drop_last=True is recommended on TPUs to maintain fixed graph shapes
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loader, test_loader, len(train_subset)

def get_model(num_classes=10):
    """
    Loads Vision Transformer (ViT-Tiny) using timm library.
    """
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
    return model

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Wrap loader with MpDeviceLoader for TPU prefetching
    para_loader = pl.MpDeviceLoader(loader, device)

    for images, labels in para_loader:
        # Note: images/labels are already moved to device by MpDeviceLoader
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # TPU-specific optimizer step (triggers XLA graph execution)
        xm.optimizer_step(optimizer)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    para_loader = pl.MpDeviceLoader(loader, device)

    with torch.no_grad():
        for images, labels in para_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

def main():
    args = get_args()
    set_seed(args.seed)

    # Acquire TPU Device
    device = xm.xla_device()
    
    # Use xm.master_print to avoid duplicate logs in distributed environments
    xm.master_print("-" * 50)
    xm.master_print(f"Configuration (TPU Optimized):")
    xm.master_print(f"  Device      : {xm.xla_real_devices([device])[0]}")
    xm.master_print(f"  Model       : ViT-Tiny (Patch16, 224x224)")
    xm.master_print(f"  Data Ratio  : {args.ratio * 100}%")
    xm.master_print(f"  Epochs      : {args.epochs}")
    xm.master_print(f"  Batch Size  : {args.batch_size}")
    xm.master_print(f"  Seed        : {args.seed}")
    xm.master_print("-" * 50)

    # Load Data
    train_loader, test_loader, n_samples = get_data_loaders(args.ratio, args.batch_size, args.num_workers)
    xm.master_print(f"Data Loaded: {n_samples} training samples.")

    # Load Model
    model = get_model()
    model = model.to(device)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import os
import numpy as np
from torch.utils.data import DataLoader, Subset

def get_args():
    parser = argparse.ArgumentParser(description='Train ResNet-18 on CIFAR-10 with Data Scaling')
    parser.add_argument('--ratio', type=float, default=1.0, help='Dataset ratio to use (0.1 to 1.0)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--workers', type=int, default=2, help='Number of data loading workers')
    return parser.parse_args()

def set_seed(seed):
    """Sets the seed for reproducibility across runs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Selects the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_data_loaders(ratio, batch_size, workers):
    """
    Prepares CIFAR-10 data loaders.
    Applies random subset sampling based on the 'ratio' argument.
    """
    # Standard augmentation for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
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
        
        # Shuffle indices with fixed seed for consistent subsets
        np.random.shuffle(indices)
        train_subset = Subset(full_trainset, indices[:split])
    else:
        train_subset = full_trainset

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, test_loader, len(train_subset)

def get_model(device):
    """
    Loads ResNet-18 and modifies the first layer for CIFAR-10 (32x32 input).
    """
    model = torchvision.models.resnet18(weights=None)
    
    # Modify initial layers to prevent information loss on small images (32x32)
    # Original ResNet is designed for ImageNet (224x224)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    
    return model.to(device)

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(loader), 100. * correct / total

def main():
    args = get_args()
    set_seed(args.seed)
    device = get_device()

    # Log Configuration
    print("-" * 50)
    print(f"Configuration:")
    print(f"  Device      : {device}")
    print(f"  Model       : ResNet-18")
    print(f"  Data Ratio  : {args.ratio * 100}%")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch Size  : {args.batch_size}")
    print(f"  Seed        : {args.seed}")
    print("-" * 50)

    # Load Data
    train_loader, test_loader, n_samples = get_data_loaders(args.ratio, args.batch_size, args.workers)
    print(f"Data Loaded: {n_samples} training samples.")

    # Load Model
    model = get_model(device)
    
    # Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training Loop
    start_time = time.time()
    best_acc = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            # torch.save(model.state_dict(), f'./models/resnet18_best.pth')

        print(f"[Epoch {epoch+1:02d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    total_time = time.time() - start_time
    
    print("-" * 50)
    print(f"Training Finished.")
    print(f"Total Time : {total_time/60:.2f} minutes")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    # Create directories if they don't exist
    if not os.path.exists('./data'):
        os.makedirs('./data')
    if not os.path.exists('./models'):
        os.makedirs('./models')
        
    main()

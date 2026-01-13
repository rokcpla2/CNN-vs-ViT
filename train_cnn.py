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
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adamw'], help='Choose optimizer type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--workers', type=int, default=2, help='Number of data loading workers')
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_data_loaders(ratio, batch_size, workers):
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

    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                 transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform_test)

    if ratio < 1.0:
        num_train = len(full_trainset)
        indices = list(range(num_train))
        split = int(np.floor(ratio * num_train))
        np.random.shuffle(indices)
        train_subset = Subset(full_trainset, indices[:split])
    else:
        train_subset = full_trainset

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, test_loader, len(train_subset)

def get_model(device):
    model = torchvision.models.resnet18(weights=None)
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

    print("-" * 50)
    print(f"Configuration:")
    print(f"  Device      : {device}")
    print(f"  Model       : ResNet-18")
    print(f"  Data Ratio  : {args.ratio * 100}%")
    print(f"  Optimizer   : {args.optimizer}")
    print(f"  LR          : {args.lr}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch Size  : {args.batch_size}")
    print("-" * 50)

    train_loader, test_loader, n_samples = get_data_loaders(args.ratio, args.batch_size, args.workers)
    print(f"Data Loaded: {n_samples} training samples.")

    model = get_model(device)

    criterion = nn.CrossEntropyLoss()

    # Optimizer 선택
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_time = time.time()
    best_acc = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc

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
    if not os.path.exists('./data'):
        os.makedirs('./data')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    main()

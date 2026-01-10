import time
import timm
import torch.nn as nn
import torch.optim as optim

# 1. 모델 생성 함수 (연구의 확장성을 위해 함수화)
def get_model(model_name, num_classes=10, pretrained=True):
    """
    model_name: 'resnet18' or 'vit_tiny_patch16_224' (ViT는 리사이즈 필요할 수 있음) 등
    pretrained: True(ImageNet 학습 가중치 사용), False(바닥부터 학습)
    """
    # CIFAR-10은 32x32이므로 ResNet은 그대로 사용 가능
    # ViT는 보통 224x224 입력을 기대하므로, 32x32용으로 설정이 조금 다를 수 있음 (timm이 자동 처리하거나 리사이즈 필요)
    # 일단 ResNet 실험을 위해 기본 설정으로 갑니다.
    
    model = timm.create_model(
        model_name, 
        pretrained=pretrained, 
        num_classes=num_classes
    )
    return model

# 2. 학습 및 평가 엔진 (Train & Evaluate Engine)
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
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
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

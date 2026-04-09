"""Seminar 07 — Pre-training (basic level).

Реактивный Marimo notebook. Запуск:
    marimo edit seminars/07-pretrain/basic.py

Цель семинара (basic):
    1. Взять предобученный ResNet-18 из torchvision.
    2. Реализовать разные стратегии fine-tuning: head-only, full, gradual.
    3. Сравнить их на подмножестве датасета.
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")

@app.cell
def __():
    import marimo as mo
    return (mo,)

@app.cell
def __(mo):
    mo.md(
        r"""
        # Семинар 07 — Fine-tuning CNN (basic)

        В этом ноутбуке мы исследуем влияние количества замороженных слоев на качество и скорость обучения.
        Мы возьмем предобученный на ImageNet ResNet-18 и дообучим его.
        """
    )
    return

@app.cell
def __():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader, Subset
    import matplotlib.pyplot as plt
    import numpy as np
    return DataLoader, Subset, datasets, models, nn, np, optim, plt, torch, transforms

@app.cell
def __(datasets, transforms, Subset, DataLoader):
    # Датасет (используем CIFAR-10 как proxy, берем только малую часть для скорости)
    transform = transforms.Compose([
        transforms.Resize(224), # ResNet требует картинки побольше
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Берем сабсет (по 100 примеров на класс для train, 50 для test)
    # Это симулирует ситуацию недостатка данных
    train_subset = Subset(train_ds, range(1000))
    test_subset = Subset(test_ds, range(500))

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    return test_ds, test_loader, test_subset, train_ds, train_loader, train_subset, transform

@app.cell
def __(mo):
    mo.md(
        r"""
        ## Выбор количества замороженных блоков

        ResNet-18 состоит из `conv1`, `layer1`, `layer2`, `layer3`, `layer4`, `fc`.
        Выберем, сколько блоков заморозить.
        """
    )
    return

@app.cell
def __(mo):
    num_frozen = mo.ui.slider(start=0, stop=5, step=1, value=4, label="Количество замороженных блоков (0 = full, 5 = head-only)")
    num_frozen
    return (num_frozen,)

@app.cell
def __(models, nn, num_frozen):
    # 1. Загружаем предобученную модель
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 2. Заменяем head на 10 классов
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    # Список блоков для заморозки в порядке от входа к выходу
    blocks = [model.conv1, model.layer1, model.layer2, model.layer3, model.layer4]
    
    # 3. Замораживаем выбранное число блоков
    frozen_count = num_frozen.value
    for i in range(frozen_count):
        for param in blocks[i].parameters():
            param.requires_grad = False
            
    # Подсчитаем параметры
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return blocks, frozen_count, model, num_ftrs, total_params, trainable_params

@app.cell
def __(model, optim, train_loader, test_loader, torch, nn):
    # Обучение
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Оптимизируем ТОЛЬКО те параметры, у которых requires_grad=True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    def train_epoch(model, dataloader, optimizer, criterion):
        model.train()
        total_loss = 0
        correct = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
        return total_loss / len(dataloader), correct / len(dataloader.dataset)
        
    def evaluate(model, dataloader, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                correct += (outputs.argmax(1) == targets).sum().item()
        return total_loss / len(dataloader), correct / len(dataloader.dataset)
        
    return criterion, device, evaluate, optimizer, train_epoch

@app.cell
def __(model, train_loader, test_loader, optimizer, criterion, train_epoch, evaluate):
    # Тренировочный цикл
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    
    print(f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    return test_acc, test_loss, train_acc, train_loss

if __name__ == "__main__":
    app.run()

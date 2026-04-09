"""Seminar NN — PyTorch (basic level).

Marimo notebook. Запуск:
    marimo edit seminars/02-pytorch/basic.py

Шаблон. Скопируйте в новый модуль и заполните клетки.
Скаффолдинг: высокий, заполняемые TODO-куски, фиксированный стек.
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
        # Семинар 02 — PyTorch Fundamentals (basic)

        Перед запуском прочитайте [лекцию](../../lectures/02-pytorch/lecture.qmd).
        Мы напишем полный цикл обучения LeNet-5 на датасете MNIST с нуля.
        """
    )
    return


@app.cell
def __():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    return DataLoader, F, datasets, nn, optim, torch, transforms


@app.cell
def __(mo):
    mo.md(r"## Гиперпараметры (Интерактивно!)")
    return


@app.cell
def __(mo):
    lr_slider = mo.ui.slider(start=0.001, stop=0.1, step=0.001, value=0.01, label="Learning Rate")
    batch_size_slider = mo.ui.slider(start=16, stop=256, step=16, value=64, label="Batch Size")
    epochs_slider = mo.ui.slider(start=1, stop=10, step=1, value=3, label="Epochs")
    
    mo.vstack([lr_slider, batch_size_slider, epochs_slider])
    return batch_size_slider, epochs_slider, lr_slider


@app.cell
def __(batch_size_slider, datasets, transforms, DataLoader):
    # Данные
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_slider.value, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return test_dataset, test_loader, train_dataset, train_loader, transform


@app.cell
def __(mo):
    mo.md(r"## Шаг 1 — Модель LeNet-5")
    return


@app.cell
def __(nn, F):
    class LeNet5(nn.Module):
        def __init__(self):
            super().__init__()
            # TODO: Реализуйте слои (Conv2d, Linear)
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            # TODO: Реализуйте прямой проход с пулингом (F.max_pool2d) и ReLU
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, 16 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
            
    return LeNet5,


@app.cell
def __(mo):
    mo.md(r"## Шаг 2 — Цикл обучения")
    return


@app.cell
def __(LeNet5, epochs_slider, lr_slider, optim, torch, train_loader, test_loader, nn):
    import matplotlib.pyplot as plt
    import io
    import marimo as mo_internal
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr_slider.value, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    
    # Мы используем tqdm для вывода в консоль, но в Marimo лучше рисовать график
    for epoch in range(epochs_slider.value):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # TODO: напишите шаги оптимизации (zero_grad, forward, backward, step)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                train_losses.append(loss.item())
                
    # Рисуем график
    fig, ax = plt.subplots()
    ax.plot(train_losses)
    ax.set_title("Training Loss")
    ax.set_xlabel("Iterations (x100)")
    ax.set_ylabel("Loss")
    
    # Оценка
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    accuracy = 100. * correct / len(test_loader.dataset)
    
    res_md = mo_internal.md(f"**Test Accuracy**: {accuracy:.2f}%")
    return accuracy, ax, correct, criterion, data, device, epoch, fig, loss, model, mo_internal, optimizer, output, pred, res_md, target, train_losses, batch_idx, plt, io


@app.cell
def __(fig, res_md, mo):
    mo.vstack([res_md, mo.as_html(fig)])
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## TODO для студента (basic)
        
        Поиграйте со слайдерами. 
        - Что будет, если поставить слишком большой Learning Rate?
        - Что будет, если сильно увеличить Batch Size?
        """
    )
    return


if __name__ == "__main__":
    app.run()

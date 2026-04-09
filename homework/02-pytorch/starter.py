"""Starter Marimo notebook for hw 02-pytorch.

Скопируйте этот файл в submission/notebook.py и работайте там. Грейдер
исполняет именно `submission/notebook.py`, не этот стартер.

Маркеры === SUBMISSION (start) === / === SUBMISSION (end) === говорят
грейдеру, какие ячейки трогать. Не удаляйте их.
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
        # ДЗ 02 — CIFAR-10 PyTorch (стартер)

        Обучите классификатор с нуля на датасете CIFAR-10.
        Не используйте `pretrained=True`.
        """
    )
    return


@app.cell
def __():
    import json
    import os
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Dataset
    return DataLoader, Dataset, Path, datasets, json, nn, np, optim, os, torch, transforms


@app.cell
def __(DataLoader, datasets, transforms):
    # Датасет CIFAR-10 для обучения
    transform_train = transforms.Compose([
        # TODO: Добавьте аугментации (RandomCrop, RandomHorizontalFlip и т.д.)
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # TODO: Настройте DataLoader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    return train_dataset, train_loader, transform_train


@app.cell
def __(Dataset, np, torch):
    class HiddenTestDataset(Dataset):
        def __init__(self, npz_path, transform=None):
            data = np.load(npz_path)
            self.images = data['images'] # Ожидается форма (N, 32, 32, 3) или (N, 3, 32, 32)
            self.transform = transform
            
            # Приводим к формату (N, 32, 32, 3) для torchvision transforms, если нужно,
            # либо обрабатываем как есть. Для простоты предполагаем (N, 32, 32, 3) uint8.
            if self.images.shape[1] == 3:
                 self.images = self.images.transpose(0, 2, 3, 1)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = self.images[idx]
            if self.transform:
                img = self.transform(img)
            return img
    return HiddenTestDataset,


@app.cell
def __():
    # === SUBMISSION (start) ===
    # Здесь — ваш код: модель, training loop.
    
    # TODO: Определите архитектуру модели
    
    # TODO: Напишите цикл обучения
    
    pass
    # === SUBMISSION (end) ===
    return


@app.cell
def __(HiddenTestDataset, DataLoader, np, os, torch, transforms):
    # Anti-cheat contract
    hidden_test_path = os.environ.get("ML3_HIDDEN_TEST", None)
    out_preds_path = os.environ.get("ML3_OUTPUT_PREDICTIONS", "submission/predictions.npy")
    
    # Имитация для локального запуска, если нет hidden_test
    if not hidden_test_path:
        print("ML3_HIDDEN_TEST not set. Please run via execution verifier or mock data locally.")
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        test_dataset = HiddenTestDataset(hidden_test_path, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # TODO: замените "model" на вашу обученную модель
        # model.eval()
        all_preds = []
        # with torch.no_grad():
        #     for inputs in test_loader:
        #         inputs = inputs.to(device)
        #         outputs = model(inputs)
        #         _, predicted = outputs.max(1)
        #         all_preds.append(predicted.cpu().numpy())
        
        # all_preds = np.concatenate(all_preds)
        
        # Заглушка, чтобы файл сохранялся:
        all_preds = np.zeros(len(test_dataset), dtype=np.int64)
        
        os.makedirs(os.path.dirname(out_preds_path), exist_ok=True)
        np.save(out_preds_path, all_preds)
        print(f"Predictions saved to {out_preds_path}")

    return all_preds, hidden_test_path, out_preds_path, test_dataset, test_loader, transform_test


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Чек-лист перед сабмитом

        - [ ] `submission/predictions.npy` существует, форма правильная.
        - [ ] `submission/notebook.py` запускается чисто на свежем `pixi run hw 02-pytorch`.
        - [ ] `submission/MODEL.md` написан под критерии из rubric.
        """
    )
    return


if __name__ == "__main__":
    app.run()

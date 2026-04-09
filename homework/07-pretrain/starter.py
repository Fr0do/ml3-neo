"""Starter Marimo notebook for hw 07-pretrain.

Скопируйте этот файл в submission/notebook.py и работайте там.
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
        # ДЗ 07 — LoRA Fine-Tuning (стартер)

        Задание исследовательское. Выберите модель, соберите данные, примените LoRA и проведите ablation study.
        """
    )
    return

@app.cell
def __():
    import torch
    import torch.nn as nn
    import numpy as np
    from pathlib import Path
    return Path, nn, np, torch

@app.cell
def __(nn, torch):
    # === SUBMISSION (start) ===
    
    # TODO 1: Реализуйте LoRALinear (или используйте библиотеку peft, если разрешено).
    # Здесь заготовка для кастомной реализации.
    class LoRALinear(nn.Module):
        def __init__(self, linear_layer: nn.Linear, rank: int, alpha: float):
            super().__init__()
            pass # TODO
            
        def forward(self, x):
            pass # TODO

    def apply_lora(model: nn.Module, target_modules: list[str], rank: int, alpha: float):
        """
        Заменяет слои в `model`, чьи имена содержатся в `target_modules`, на `LoRALinear`.
        """
        pass # TODO
        
    return LoRALinear, apply_lora

@app.cell
def __(mo):
    mo.md(r"### Блок для подсчета параметров (Anti-cheat)")
    return

@app.cell
def __():
    # TODO 2: Напишите код для подсчета обучаемых параметров в вашей модели.
    # Выведите в stdout строку вида: "Trainable params: X || All params: Y || Trainable%: Z"
    pass
    return

@app.cell
def __():
    # TODO 3: Обучение модели (train loop) и предсказания.
    
    # ML3_HIDDEN_TEST
    # predictions = ... # ваш код получения вероятностей/меток (N,) на тестовом/валидационном сете
    # ML3_OUTPUT_PREDICTIONS
    # np.save("submission/predictions.npy", predictions)
    pass
    # === SUBMISSION (end) ===
    return

if __name__ == "__main__":
    app.run()

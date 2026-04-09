"""Seminar 07 — Pre-training (advanced level).

Реактивный Marimo notebook. Запуск:
    marimo edit seminars/07-pretrain/advanced.py

Цель семинара (advanced):
    1. Написать LoRALinear wrapper с нуля.
    2. Интегрировать его в небольшую языковую модель.
    3. Провести ablation study по рангу $r$.
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
        # Семинар 07 — LoRA с нуля (advanced)

        Вместо использования готовой библиотеки `peft`, мы реализуем `LoRALinear` сами.
        Это позволит понять, как математика $\Delta W = BA$ реализуется в коде.
        """
    )
    return

@app.cell
def __():
    import torch
    import torch.nn as nn
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    return math, nn, np, plt, torch

@app.cell
def __(nn, math, torch):
    class LoRALinear(nn.Module):
        """
        Обертка над обычным nn.Linear, реализующая логику LoRA.
        Формула: output = linear(x) + (x @ A.T @ B.T) * scaling
        """
        def __init__(self, linear_layer: nn.Linear, rank: int, alpha: float):
            super().__init__()
            self.linear = linear_layer # Оригинальный слой
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            
            # Замораживаем оригинальные веса
            self.linear.weight.requires_grad = False
            if self.linear.bias is not None:
                self.linear.bias.requires_grad = False
                
            in_features = self.linear.in_features
            out_features = self.linear.out_features
            
            # Инициализируем матрицы A и B
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            
            self.reset_parameters()
            
        def reset_parameters(self):
            # Инициализация A (нормальное распределение, kaiming) и B (нули)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = self.linear(x)
            lora_out = (x @ self.lora_A.T) @ self.lora_B.T
            return base_out + lora_out * self.scaling
            
        def merge_weights(self):
            """Слияние весов для быстрого инференса"""
            delta_W = (self.lora_B @ self.lora_A) * self.scaling
            self.linear.weight.data += delta_W
            
    return LoRALinear,

@app.cell
def __(mo):
    mo.md(
        r"""
        ## Ablation Study по рангу $r$
        """
    )
    return

@app.cell
def __(LoRALinear, nn, plt, np):
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    ranks = [1, 2, 4, 8, 16, 32, 64]
    params_count = []
    
    d_model = 768
    
    for r in ranks:
        base_linear = nn.Linear(d_model, d_model)
        lora_layer = LoRALinear(base_linear, rank=r, alpha=32)
        params_count.append(count_parameters(lora_layer))
        
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ranks, params_count, marker="o")
    ax.set_xlabel("LoRA Rank ($r$)")
    ax.set_ylabel("Trainable Parameters")
    ax.set_title("Зависимость числа параметров от ранга (для слоя 768x768)")
    ax.grid(True)
    
    ax
    return ax, count_parameters, d_model, fig, lora_layer, params_count, r, ranks, base_linear

if __name__ == "__main__":
    app.run()

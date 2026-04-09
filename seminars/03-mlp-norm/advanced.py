"""Seminar 03 — MLP & Norms (advanced level).

Реактивный Marimo notebook. Запуск:
    marimo edit seminars/03-mlp-norm/advanced.py

Открытая задача:
    1. Реализовать GroupNorm и RMSNorm с нуля.
    2. Провести Ablation Study на CIFAR-10 (3 архитектуры x 5 нормализаций).
    3. Построить таблицу accuracy + training_time и объяснить паттерны.
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
        # Семинар 03 — Нормализации (advanced)

        Открытая задача:
        1. Реализуйте `GroupNorm` и `RMSNorm`.
        2. Запустите ablation-эксперимент на подмножестве CIFAR-10 или табличных данных.
        3. Сравните: Без нормализации, BN, LN, GN, RMSNorm по метрикам `accuracy` и `training_time`.
        """
    )
    return


@app.cell
def __():
    import torch
    import torch.nn as nn
    return nn, torch


@app.cell
def __(nn, torch):
    class CustomGroupNorm(nn.Module):
        def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
            super().__init__()
            assert num_channels % num_groups == 0
            # TODO: Реализовать GroupNorm
            pass
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # TODO: Реализовать forward
            return x
    return (CustomGroupNorm,)


@app.cell
def __(nn, torch):
    class CustomRMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            # TODO: Реализовать RMSNorm
            pass
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # TODO: Реализовать forward (без центрирования!)
            return x
    return (CustomRMSNorm,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Ablation Study

        Ниже реализуйте тренировочный цикл. Попробуйте разные архитектуры (например: MLP 3 слоя, MLP 5 слоев, CNN 3 слоя).
        Соберите результаты в Pandas DataFrame и выведите через `mo.ui.table`.
        Объясните, почему для CNN лучше подошел один метод, а для MLP - другой.
        """
    )
    return


if __name__ == "__main__":
    app.run()

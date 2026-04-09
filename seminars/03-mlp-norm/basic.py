"""Seminar 03 — MLP & Norms (basic level).

Реактивный Marimo notebook. Запуск:
    marimo edit seminars/03-mlp-norm/basic.py

Цель семинара (basic):
    1. Реализовать BatchNorm1d и LayerNorm с нуля.
    2. Сравнить с official impl на синтетике.
    3. Показать разницу train/eval для BN и нестабильность при малых батчах.
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
        # Семинар 03 — MLP и Нормализации (basic)

        В этом семинаре мы своими руками напишем `BatchNorm1d` и `LayerNorm` 
        используя только базовые тензорные операции PyTorch.
        
        Затем мы проверим их на синтетических данных и посмотрим на главную 
        проблему `BatchNorm` — нестабильность при маленьком `batch_size`.
        """
    )
    return


@app.cell
def __():
    import torch
    import torch.nn as nn
    return nn, torch


@app.cell
def __(mo):
    batch_size = mo.ui.slider(2, 128, value=32, step=2, label="Batch size")
    mo.hstack([batch_size])
    return (batch_size,)


@app.cell
def __(nn, torch):
    class CustomBatchNorm1d(nn.Module):
        def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
            super().__init__()
            self.eps = eps
            self.momentum = momentum
            
            # Обучаемые параметры (affine)
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
            
            # Running stats (не обучаются градиентным спуском)
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.training:
                # Усреднение по батчу (dim=0)
                mean = x.mean(dim=0)
                # unbiased=False для совпадения с PyTorch BatchNorm
                var = x.var(dim=0, unbiased=False)
                
                # Обновление скользящих средних
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * (x.var(dim=0, unbiased=True))
            else:
                mean = self.running_mean
                var = self.running_var
                
            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            return self.gamma * x_hat + self.beta
    return (CustomBatchNorm1d,)


@app.cell
def __(nn, torch):
    class CustomLayerNorm(nn.Module):
        def __init__(self, normalized_shape: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Усреднение по каналам/фичам (dim=-1), keepdim=True для броадкаста
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, unbiased=False, keepdim=True)
            
            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            return self.gamma * x_hat + self.beta
    return (CustomLayerNorm,)


@app.cell
def __(CustomBatchNorm1d, batch_size, mo, nn, torch):
    # Тестируем реализацию BN
    features = 16
    x = torch.randn(batch_size.value, features) * 5.0 + 2.0
    
    my_bn = CustomBatchNorm1d(features)
    pt_bn = nn.BatchNorm1d(features)
    
    my_out = my_bn(x)
    pt_out = pt_bn(x)
    
    diff_train = (my_out - pt_out).abs().max().item()
    
    my_bn.eval()
    pt_bn.eval()
    x_test = torch.randn(batch_size.value, features)
    
    diff_eval = (my_bn(x_test) - pt_bn(x_test)).abs().max().item()
    
    mo.md(f"""
    **Сравнение CustomBatchNorm1d и nn.BatchNorm1d:**
    - Train diff: `{diff_train:.6f}`
    - Eval diff: `{diff_eval:.6f}`
    - Batch size: `{batch_size.value}`
    """)
    return diff_eval, diff_train, features, my_bn, my_out, pt_bn, pt_out, x, x_test


@app.cell
def __(mo):
    mo.md(
        r"""
        ## TODO для студента

        1. Измени слайдер `batch_size` на значение `2`. Посмотри на train и eval diff.
        2. Почему при малом батче BatchNorm работает нестабильно?
        3. Напиши тесты сравнения твоего `CustomLayerNorm` с `nn.LayerNorm`.
        4. Что произойдет, если в LayerNorm поменять `batch_size` на `2`? 
           Изменится ли стабильность? (Подсказка: нет, проверь это).
        """
    )
    return


if __name__ == "__main__":
    app.run()

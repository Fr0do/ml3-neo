"""Seminar NN — PyTorch (advanced level).

Шаблон. Скаффолдинга минимум, открытая постановка, ссылка на ДЗ как на
естественное продолжение.
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
        # Семинар 02 — PyTorch (advanced)

        ## Задача

        Реализуйте оптимизированный цикл обучения. Ваша задача — ускорить обучение и повысить стабильность.
        
        Требования:
        1. **Gradient Accumulation**: реализуйте эффективный батч сайз больше, чем влезает в память (например, копите градиенты 4 шага).
        2. **Mixed Precision (AMP)**: добавьте `torch.cuda.amp.autocast` и `GradScaler` для ускорения на современных GPU.
        3. **LR Scheduler**: добавьте `CosineAnnealingLR`.
        4. **Profiler**: оберните обучение в `torch.profiler.profile` и найдите bottleneck в вашем pipeline.

        ## Чек-лист реализации

        - [ ] Свой цикл с градиентной аккумуляцией
        - [ ] Интеграция AMP
        - [ ] Cosine Annealing работает (LR уменьшается)
        - [ ] Вывод throughput (samples/sec) и таблицы профилировщика
        """
    )
    return


@app.cell
def __():
    # === SUBMISSION (start) ===
    # Здесь — ваш код. Грейдер исполняет всё между маркерами.
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    # TODO: Имплементация
    
    # === SUBMISSION (end) ===
    return DataLoader, datasets, nn, optim, torch, transforms


if __name__ == "__main__":
    app.run()

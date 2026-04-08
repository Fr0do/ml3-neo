"""Seminar 04 — CNN (advanced level).

Реактивный Marimo notebook. Запуск:
    marimo edit seminars/04-cnn/advanced.py

В отличие от basic.py, здесь почти нет скаффолдинга. Цель — собрать
ResNet-18 с нуля, сравнить с эталоном из torchvision и провести три
эксперимента, описанных в финальной ячейке. Все ответы — кодом и
визуализациями, не текстом.
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
        # Семинар 04 — CNN (advanced)

        ## Правила игры

        1. **Никаких готовых ResNet'ов из torchvision** в финальном решении —
           только как reference для проверки.
        2. Все эксперименты — *воспроизводимы*: фиксируешь seed, в Marimo
           пересчёт сразу видно.
        3. Все графики — встроены в этот ноутбук, не в отдельных файлах.

        ## Задача

        Соберите ResNet-18 BasicBlock'ами с **двумя версиями нормализации**:
        BatchNorm и GroupNorm. Обучите обе на CIFAR-10 с одинаковыми
        гиперпараметрами при разных размерах батча: $\{4, 16, 64, 256\}$.

        Постройте таблицу test accuracy и постарайтесь объяснить наблюдаемый
        кроссовер между BN и GN при уменьшении батча.
        """
    )
    return


@app.cell
def __():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return F, nn, torch


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Чек-лист реализации

        - [ ] `BasicBlock(in_c, out_c, stride, norm_layer)` — параметризован
              normalize-фабрикой.
        - [ ] `ResNet18(num_classes, norm_layer)` — 4 stage по 2 блока.
        - [ ] Сравнение с `torchvision.models.resnet18(weights=None)` по числу
              параметров и shapes на одном входе.
        - [ ] Training loop с фиксированным seed.
        - [ ] Эксперимент 1: BN vs GN при `batch_size in {4, 16, 64, 256}`.
        - [ ] Эксперимент 2: добавить аугментации (Albumentations) — выигрыш?
        - [ ] Эксперимент 3: receptive field stage1..stage4 через impulse-трюк.
        """
    )
    return


@app.cell
def __():
    # === SUBMISSION (start) ===
    # Здесь вы пишете BasicBlock и ResNet18.
    # Грейдер исполняет всё, что между маркерами === SUBMISSION ===.
    pass
    # === SUBMISSION (end) ===
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Что сдавать

        Эта версия семинара — *тренировка* перед ДЗ
        [`04-cnn-zoo`](../../homework/04-cnn-zoo/README.qmd). На семинаре
        ничего сдавать не нужно, но именно те же эксперименты вам понадобятся
        в ДЗ как baseline.

        Если за семинар получилось дойти только до сравнения BN vs GN при
        одном размере батча — это нормально, остальные два эксперимента
        делаем дома.
        """
    )
    return


if __name__ == "__main__":
    app.run()

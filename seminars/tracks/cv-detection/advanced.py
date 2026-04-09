"""Seminar CV — Detection (advanced level).

Optional track: cv-detection.
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
        # Семинар CV: Detection (advanced)

        ## Задача

        Реализуйте упрощённый детектор (single-class, single-scale YOLO-style) 
        на синтетических данных (квадраты/круги на случайном фоне). 
        Покажите шаг NMS.
        Визуализируйте predicted boxes vs ground truth.

        ## Чек-лист реализации

        - [ ] Сгенерировать синтетический датасет: картинки $H \times W$, на которых случайно разбросаны квадраты.
        - [ ] Определить YOLO-like архитектуру: $S \times S$ grid, предсказание $(x, y, w, h, C)$.
        - [ ] Написать Loss: MSE для координат + BCE для confidence.
        - [ ] Написать шаг NMS (Non-Maximum Suppression).
        - [ ] Отрисовать предсказания до и после NMS.
        """
    )
    return

@app.cell
def __():
    # === SUBMISSION (start) ===
    import torch
    import torch.nn as nn
    import torchvision.ops as ops
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # TODO: Реализовать генерацию данных
    # TODO: Реализовать SimpleYOLO
    # TODO: Написать training loop
    # TODO: Применить ops.nms(boxes, scores, iou_threshold)
    # TODO: Написать функцию визуализации
    pass
    # === SUBMISSION (end) ===
    return

if __name__ == "__main__":
    app.run()

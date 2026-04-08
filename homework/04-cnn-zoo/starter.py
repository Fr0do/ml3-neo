"""Starter Marimo notebook for hw 04-cnn-zoo.

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
        # ДЗ 04 — cnn-zoo (стартер)

        TinyImageNet, 40 классов, 64×64. Цель — обучить классификатор и
        записать предсказания в `submission/predictions.npy`.

        Прочитайте [README](README.qmd) и [`rubric.yml`](rubric.yml) до того,
        как начать. Rubric — открытый, под него можно оптимизировать
        `MODEL.md`.
        """
    )
    return


@app.cell
def __():
    import json
    from pathlib import Path

    import numpy as np
    import torch
    return Path, json, np, torch


@app.cell
def __(Path, np):
    # Загружаем публичный train/dev сет.
    DATA = Path("homework/04-cnn-zoo/data/public")
    train = np.load(DATA / "train.npz") if (DATA / "train.npz").exists() else None
    dev = np.load(DATA / "dev.npz") if (DATA / "dev.npz").exists() else None
    test_index = (DATA / "test_index.json")
    return DATA, dev, test_index, train


@app.cell
def __():
    # === SUBMISSION (start) ===
    # Здесь — ваш код: модель, training loop, инференс.
    # Финальная инструкция: записать predictions в submission/predictions.npy
    # с формой (N,) и dtype int64, где N — длина test_index.json.
    pass
    # === SUBMISSION (end) ===
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Чек-лист перед сабмитом

        - [ ] `submission/predictions.npy` существует, форма правильная.
        - [ ] `submission/notebook.py` запускается чисто на свежем `pixi run hw 04-cnn-zoo`.
        - [ ] `submission/MODEL.md` написан под все 6 критериев из rubric.
        - [ ] Никаких изменений в `eval.py`, `tests/`, `meta.yml`, `rubric.yml`.
        - [ ] Локально прогнан `pytest homework/04-cnn-zoo/tests/`.
        """
    )
    return


if __name__ == "__main__":
    app.run()

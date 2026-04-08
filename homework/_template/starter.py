"""Стартер Marimo notebook для нового ДЗ. Скопируйте и адаптируйте."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(r"# ДЗ <id> — <название> (стартер)")
    return


@app.cell
def __():
    # === SUBMISSION (start) ===
    # Контракт execution verifier'а:
    #   ML3_HIDDEN_TEST       — путь к вашему персональному test.npz
    #   ML3_OUTPUT_PREDICTIONS — куда писать predictions.npy
    #
    # Локально в Marimo эти переменные отсутствуют — тогда используйте
    # публичный dev-set из data/public/. Грейдер подставит их сам.
    import os
    import numpy as np

    hidden_path = os.environ.get("ML3_HIDDEN_TEST")
    out_path = os.environ.get("ML3_OUTPUT_PREDICTIONS")

    if hidden_path and out_path:
        test = np.load(hidden_path)
        # TODO: загрузить модель, получить предсказания на test['images']
        # preds = model(test['images']).argmax(-1).cpu().numpy()
        preds = np.zeros(len(test["labels"]), dtype=np.int64)
        np.save(out_path, preds)
    # === SUBMISSION (end) ===
    return


if __name__ == "__main__":
    app.run()

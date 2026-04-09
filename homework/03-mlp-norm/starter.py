"""Starter Marimo notebook for hw 03-mlp-norm.

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
        # ДЗ 03 — Нормализации и Ablation Study

        Цель — провести чистое сравнение различных методов нормализации.
        Заполните код экспериментов ниже и сохраните лучшие предсказания.
        """
    )
    return


@app.cell
def __():
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    return nn, np, os, torch


@app.cell
def __(os, np):
    # Анти-чит окружение для грейдера
    HIDDEN_TEST = os.environ.get("ML3_HIDDEN_TEST", None)
    OUTPUT_PREDICTIONS = os.environ.get("ML3_OUTPUT_PREDICTIONS", "submission/predictions.npy")
    
    # Загрузите ваши данные для обучения (train/dev)
    # Если HIDDEN_TEST задан, используйте его для формирования финальных предсказаний
    return HIDDEN_TEST, OUTPUT_PREDICTIONS


@app.cell
def __():
    def run_experiment(norm_type: str, model_config: dict, train_data, val_data) -> dict:
        """
        Универсальная функция для запуска одного эксперимента (ablation).
        Должна возвращать словарь с метриками (например, val_accuracy, train_time).
        """
        # TODO: Реализовать цикл обучения с фиксацией seed
        metrics = {"val_accuracy": 0.0, "norm_type": norm_type}
        return metrics
    return (run_experiment,)


@app.cell
def __(mo):
    mo.md("## Сравнение нормализаций")
    return


@app.cell
def __(run_experiment):
    # === SUBMISSION (start) ===
    # Запустите ablations
    
    # results_bn = run_experiment("BatchNorm", ...)
    # results_ln = run_experiment("LayerNorm", ...)
    
    # Соберите финальную модель, обучите и сделайте предсказания на тесте (или HIDDEN_TEST)
    # Сохраните в OUTPUT_PREDICTIONS (с формой (N,) и dtype int64)
    # np.save(OUTPUT_PREDICTIONS, final_preds)
    pass
    # === SUBMISSION (end) ===
    return


if __name__ == "__main__":
    app.run()

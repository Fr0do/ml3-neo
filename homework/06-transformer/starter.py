"""Starter Marimo notebook for hw 06-transformer.

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
        # ДЗ 06 — transformer (стартер)

        Реализуйте блок Transformer Encoder с нуля и примените его к классификации текстов.
        """
    )
    return


@app.cell
def __():
    import os
    import json
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    return Path, json, nn, np, os, torch


@app.cell
def __(Path, os, np):
    # Данные для обучения
    DATA = Path("homework/06-transformer/data/public")
    train_path = DATA / "train.npz"
    train = np.load(train_path) if train_path.exists() else None
    
    # Чтение окружения для грейдера
    hidden_test_path = os.environ.get("ML3_HIDDEN_TEST", str(DATA / "test.npz"))
    out_preds_path = os.environ.get("ML3_OUTPUT_PREDICTIONS", "submission/predictions.npy")
    return DATA, hidden_test_path, out_preds_path, train, train_path


@app.cell
def __(nn):
    # === SUBMISSION (start) ===
    class SelfAttention(nn.Module):
        # TODO: Реализовать Scaled Dot-Product Attention
        pass

    class MultiHeadAttention(nn.Module):
        # TODO: Реализовать Multi-Head Attention
        pass
        
    class TransformerBlock(nn.Module):
        # TODO: Реализовать блок (Pre-LN или Post-LN)
        pass
        
    class TransformerClassifier(nn.Module):
        # TODO: Собрать финальную модель (с Embedding, Positional Encoding, слоями, [CLS] или pooling)
        pass

    # TODO: Обучить модель на train данных, получить предсказания для hidden_test
    # predictions = ...
    
    # Сохраняем результат
    # import os
    # import numpy as np
    # out_path = os.environ.get("ML3_OUTPUT_PREDICTIONS", "submission/predictions.npy")
    # os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # np.save(out_path, predictions)
    pass
    # === SUBMISSION (end) ===
    return (
        MultiHeadAttention,
        SelfAttention,
        TransformerBlock,
        TransformerClassifier,
    )


if __name__ == "__main__":
    app.run()

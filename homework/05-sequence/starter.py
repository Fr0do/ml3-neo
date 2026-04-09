"""Starter Marimo notebook for hw 05-sequence.

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
        # ДЗ 05 — Классификация текстов (стартер)

        Обучите модель классификации текстов (LSTM/GRU + Attention).
        Сохраните результаты в `submission/predictions.npy`.
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
    from torch.utils.data import Dataset, DataLoader
    return DataLoader, Dataset, Path, json, nn, np, os, torch


@app.cell
def __(Dataset, Path, np, os):
    # Anti-cheat / Hidden test check
    # Грейдер устанавливает ML3_HIDDEN_TEST
    HIDDEN_TEST = os.environ.get("ML3_HIDDEN_TEST")

    if HIDDEN_TEST and Path(HIDDEN_TEST).exists():
        print("Running in grading mode on hidden test!")
        test_data_path = Path(HIDDEN_TEST)
    else:
        print("Running in local mode on public test.")
        test_data_path = Path("homework/05-sequence/data/public/test.npz")
        
    # Dummy Dataset для примера
    class TextDataset(Dataset):
        def __init__(self, texts, labels=None):
            self.texts = texts
            self.labels = labels
            
        def __len__(self):
            return len(self.texts)
            
        def __getitem__(self, idx):
            item = {"text": self.texts[idx]}
            if self.labels is not None:
                item["label"] = self.labels[idx]
            return item
    return HIDDEN_TEST, TextDataset, test_data_path


@app.cell
def __():
    class TokenizerSimple:
        def __init__(self, char_level=True):
            self.char_level = char_level
            self.vocab = {"<PAD>": 0, "<UNK>": 1}
            
        def fit(self, texts):
            pass # TODO: собрать словарь
            
        def encode(self, text):
            return [1] # TODO: вернуть токены
    return TokenizerSimple,


@app.cell
def __(nn):
    class LSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            # TODO: Add Attention here
            self.fc = nn.Linear(hidden_dim, num_classes)
            
        def forward(self, x):
            embedded = self.embedding(x)
            out, (hn, cn) = self.lstm(embedded)
            # Базовый вариант без attention: используем последнее скрытое состояние
            logits = self.fc(hn[-1])
            return logits
    return LSTMClassifier,


@app.cell
def __():
    # === SUBMISSION (start) ===
    # ML3_HIDDEN_TEST
    
    # predictions = ...
    # np.save("submission/predictions.npy", predictions)
    # ML3_OUTPUT_PREDICTIONS
    pass
    # === SUBMISSION (end) ===
    return


if __name__ == "__main__":
    app.run()

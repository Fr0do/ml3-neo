"""Seminar 06 — Transformer (advanced level).

Открытая задача: реализовать мини-GPT (decoder-only) и обучить на корпусе текстов
(Project Gutenberg).

Запуск:
    marimo edit seminars/06-transformer/advanced.py
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
        # Семинар 06 — Transformer (advanced)

        Открытая задача:
        1. Реализовать `CausalSelfAttention` (с маскированием будущего).
        2. Реализовать `TransformerDecoderBlock`.
        3. Сравнить стабильность `Pre-LN` и `Post-LN`.
        4. Скачать текст (например, [Pushkin](https://gutenberg.org/cache/epub/2238/pg2238.txt) из Project Gutenberg).
        5. Обучить character-level GPT (начни с 2 слоёв, потом попробуй 12).
        6. Сгенерировать текст (temperature, top-k, top-p sampling).
        """
    )
    return


@app.cell
def __():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import urllib.request
    return F, nn, torch, urllib.request


@app.cell
def __(mo):
    mo.md("Напиши здесь свою реализацию мини-GPT!")
    return


if __name__ == "__main__":
    app.run()

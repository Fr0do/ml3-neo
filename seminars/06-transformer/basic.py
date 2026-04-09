"""Seminar 06 — Transformer (basic level).

Реактивный Marimo notebook. Запуск:
    marimo edit seminars/06-transformer/basic.py

Цель семинара (basic):
    1. Реализовать Scaled Dot-Product Attention.
    2. Реализовать Multi-Head Attention с нуля.
    3. Визуализировать attention weights (heatmap).
    4. Применить для классификации последовательности.
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
        # Семинар 06 — Transformer (basic)

        Перед запуском убедитесь, что прочитали [лекцию](../../lectures/06-transformer/lecture.qmd).

        Ниже мы:
        1. Напишем `ScaledDotProductAttention`.
        2. Сделаем из него `MultiHeadAttention`.
        3. Применим к задаче sequence classification на синтетических данных.
        4. Построим heatmap для attention weights.
        """
    )
    return


@app.cell
def __():
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np
    return F, math, nn, plt, np, torch


@app.cell
def __(F, math, nn, torch):
    class ScaledDotProductAttention(nn.Module):
        def forward(self, q, k, v, mask=None):
            # q, k, v: [batch_size, n_heads, seq_len, head_dim]
            d_k = q.size(-1)
            
            # TODO: вычислить scores = Q * K^T / sqrt(d_k)
            # scores = ...
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            # TODO: вычислить weights и output
            # weights = ...
            # output = ...
            weights = F.softmax(scores, dim=-1)
            output = torch.matmul(weights, v)
            
            return output, weights
    return (ScaledDotProductAttention,)


@app.cell
def __(ScaledDotProductAttention, nn, torch):
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int):
            super().__init__()
            assert d_model % n_heads == 0
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            
            self.q_linear = nn.Linear(d_model, d_model)
            self.k_linear = nn.Linear(d_model, d_model)
            self.v_linear = nn.Linear(d_model, d_model)
            self.out_linear = nn.Linear(d_model, d_model)
            self.attention = ScaledDotProductAttention()
            
        def forward(self, q, k, v, mask=None):
            bs = q.size(0)
            
            # Linear projections & reshape to [batch_size, n_heads, seq_len, d_k]
            q = self.q_linear(q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
            k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
            v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
            
            scores, weights = self.attention(q, k, v, mask)
            
            # Concat heads and project
            concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
            output = self.out_linear(concat)
            
            return output, weights
    return (MultiHeadAttention,)


@app.cell
def __(MultiHeadAttention, mo, torch):
    n_heads_ui = mo.ui.slider(1, 8, value=4, step=1, label="n_heads")
    mo.md(f"Выберите количество голов: {n_heads_ui}")
    return (n_heads_ui,)


@app.cell
def __(MultiHeadAttention, nn, torch):
    class TransformerClassifierSynthetic(nn.Module):
        def __init__(self, vocab_size, d_model, n_heads, num_classes):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d_model)
            self.mha = MultiHeadAttention(d_model, n_heads)
            self.fc = nn.Linear(d_model, num_classes)
            
        def forward(self, x):
            # x: [batch_size, seq_len]
            x = self.emb(x)
            # Применяем внимание
            attended, weights = self.mha(x, x, x)
            # Pooling (average)
            pooled = attended.mean(dim=1)
            out = self.fc(pooled)
            return out, weights
    return (TransformerClassifierSynthetic,)


@app.cell
def __(TransformerClassifierSynthetic, n_heads_ui, plt, torch):
    # Тест классификации и визуализация attention
    torch.manual_seed(42)
    seq_len = 10
    d_model = 64
    vocab_size = 100
    num_classes = 2
    
    model = TransformerClassifierSynthetic(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_heads=n_heads_ui.value, 
        num_classes=num_classes
    )
    
    x = torch.randint(0, vocab_size, (1, seq_len))
    out, weights = model(x)
    
    fig, axes = plt.subplots(1, n_heads_ui.value, figsize=(3 * n_heads_ui.value, 3))
    if n_heads_ui.value == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        head_weights = weights[0, i].detach().numpy()
        im = ax.imshow(head_weights, cmap="viridis")
        ax.set_title(f"Head {i+1}")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
    
    plt.tight_layout()
    return axes, d_model, fig, head_weights, i, im, model, num_classes, out, seq_len, vocab_size, weights, x


@app.cell
def __(fig, mo):
    mo.center(fig)
    return


if __name__ == "__main__":
    app.run()

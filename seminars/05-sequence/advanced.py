"""Seminar NN — Sequence Models (advanced level).

Marimo notebook. Запуск:
    marimo edit seminars/05-sequence/advanced.py
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
        # Семинар NN — Sequence models (advanced)
        
        Реализация Bahdanau Attention.
        """
    )
    return


@app.cell
def __():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import plotly.express as px
    import numpy as np
    return F, nn, np, px, torch


@app.cell
def __(F, nn, torch):
    class BahdanauAttention(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.W = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Linear(hidden_size, 1, bias=False)

        def forward(self, hidden, encoder_outputs):
            # hidden: (batch, hidden_size) - decoder state
            # encoder_outputs: (batch, seq_len, hidden_size)
            
            # TODO: реализовать подсчет alignment scores и контекстного вектора
            seq_len = encoder_outputs.size(1)
            hidden_expanded = hidden.unsqueeze(1).repeat(1, seq_len, 1)
            
            energy = torch.tanh(self.W(torch.cat([hidden_expanded, encoder_outputs], dim=2)))
            attention_scores = self.v(energy).squeeze(2) # (batch, seq_len)
            
            attention_weights = F.softmax(attention_scores, dim=1)
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            
            return context, attention_weights
    return BahdanauAttention,


@app.cell
def __(mo):
    mo.md(r"## Визуализация Alignment Matrix")
    return


@app.cell
def __(BahdanauAttention, np, px, torch):
    # Dummy weights for visualization
    seq_len_src = 15
    seq_len_tgt = 10
    dummy_weights = np.random.rand(seq_len_tgt, seq_len_src)
    dummy_weights = dummy_weights / dummy_weights.sum(axis=1, keepdims=True)
    
    fig = px.imshow(dummy_weights, labels=dict(x="Source words", y="Target words", color="Attention"),
                    title="Alignment Matrix Visualization")
    fig
    return dummy_weights, fig, seq_len_src, seq_len_tgt


@app.cell
def __(mo):
    mo.md(r"## TODO для студента (advanced): Обучить seq2seq с Attention на реверс строки и построить тепловую карту для обученной модели.")
    return


if __name__ == "__main__":
    app.run()

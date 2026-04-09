"""Seminar NN — Sequence Models (basic level).

Marimo notebook. Запуск:
    marimo edit seminars/05-sequence/basic.py
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
        # Семинар NN — Sequence models (basic)

        Перед запуском прочитайте [лекцию](../../lectures/05-sequence/lecture.qmd).
        """
    )
    return


@app.cell
def __():
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    return nn, plt, torch


@app.cell
def __(mo):
    mo.md(r"## Шаг 1 — Реализация RNNCell")
    return


@app.cell
def __(nn, torch):
    class CustomRNNCell(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
            self.W_hh = nn.Linear(hidden_size, hidden_size)
            self.tanh = nn.Tanh()

        def forward(self, x, h):
            # x: (batch, input_size)
            # h: (batch, hidden_size)
            return self.tanh(self.W_xh(x) + self.W_hh(h))
    return CustomRNNCell,


@app.cell
def __(mo):
    mo.md(r"## Шаг 2 — Сравнение RNN и LSTM на длинных последовательностях")
    return


@app.cell
def __(mo):
    seq_length_slider = mo.ui.slider(start=10, stop=100, step=10, value=20, label="Sequence Length")
    seq_length_slider
    return seq_length_slider,


@app.cell
def __(CustomRNNCell, nn, seq_length_slider, torch):
    # Dummy data generation and training loop comparison
    seq_len = seq_length_slider.value
    batch_size = 32
    input_dim = 10
    hidden_dim = 20
    
    # Сравним сходимость (в реальном семинаре здесь будет training loop)
    # Покажем инициализацию моделей:
    my_rnn_cell = CustomRNNCell(input_dim, hidden_dim)
    lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    
    x = torch.randn(batch_size, seq_len, input_dim)
    out_lstm, _ = lstm(x)
    print(f"Sequence length: {seq_len}")
    print(f"LSTM output shape: {out_lstm.shape}")
    return batch_size, hidden_dim, input_dim, lstm, my_rnn_cell, out_lstm, seq_len, x


@app.cell
def __(mo):
    mo.md(r"## TODO для студента (basic): реализовать полный проход с CustomRNNCell по seq_len")
    return


if __name__ == "__main__":
    app.run()

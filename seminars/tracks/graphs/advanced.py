"""Seminar Graphs — GNN (advanced level)."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import torch
    import torch.nn as nn
    return mo, torch, nn


@app.cell
def __(mo):
    mo.md(
        r"""
        # Семинар Graphs — GNN (advanced)

        ## Задача

        1. Реализовать GCN с нуля (только torch, без PyG).
        2. Применить к node classification на Karate Club датасете.
        3. Визуализировать learned embeddings (PCA 2D).
        4. Сравнить с GAT.

        ## Чек-лист реализации

        - [ ] GCN Layer math
        - [ ] Karate Club data load
        - [ ] Training loop
        - [ ] PCA visualization
        """
    )
    return


@app.cell
def __(torch, nn):
    # === SUBMISSION (start) ===
    class GCNLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            
        def forward(self, x, adj):
            # TODO: implement D^{-1/2} A D^{-1/2} X W
            support = self.linear(x)
            out = torch.matmul(adj, support)
            return out

    # TODO: training on Karate Club, PCA
    # === SUBMISSION (end) ===
    return GCNLayer,


if __name__ == "__main__":
    app.run()

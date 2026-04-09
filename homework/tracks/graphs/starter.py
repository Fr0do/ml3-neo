import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")

@app.cell
def __():
    import marimo as mo
    import torch.nn as nn
    return mo, nn

@app.cell
def __(mo):
    mo.md("# Homework Graphs")
    return

@app.cell
def __(nn):
    class GCNLayer(nn.Module):
        def __init__(self):
            super().__init__()
            # TODO
            pass

    class GraphDataset:
        # TODO
        pass
        
    # ANTI-CHEAT BLOCK
    # ...
    return GCNLayer, GraphDataset

if __name__ == "__main__":
    app.run()

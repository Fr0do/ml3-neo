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
    mo.md("# Homework RL")
    return

@app.cell
def __(nn):
    class PolicyNet(nn.Module):
        # TODO
        pass

    class ValueNet(nn.Module):
        # TODO
        pass
        
    def compute_returns(rewards, gamma):
        # TODO
        pass
        
    def training_loop():
        # TODO
        pass

    return PolicyNet, ValueNet, compute_returns, training_loop

if __name__ == "__main__":
    app.run()

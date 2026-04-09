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
    mo.md("# Homework Generative")
    return

@app.cell
def __(nn):
    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            pass
            
    def ddpm_forward():
        # TODO
        pass
        
    def ddpm_reverse_cfg():
        # TODO: CFG
        pass
    return UNet, ddpm_forward, ddpm_reverse_cfg

if __name__ == "__main__":
    app.run()

"""Starter Marimo notebook for hw 08-generative.

Скопируйте этот файл в submission/notebook.py и работайте там.
Грейдер исполняет именно `submission/notebook.py`.
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
        # ДЗ 08 — VAE и Латентное пространство

        Вам предстоит реализовать VAE, обучить его и исследовать латентное пространство.
        Прочитайте `README.qmd` и `rubric.yml` перед началом.
        """
    )
    return

@app.cell
def __():
    import json
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    return Path, json, nn, np, plt, torch

@app.cell
def __():
    # === SUBMISSION (start) ===
    # TODO: Реализуйте Encoder, Decoder и VAE
    
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            pass
            
        def forward(self, x):
            # Верните mu, logvar
            pass
            
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            pass
            
        def forward(self, z):
            pass

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            pass
            
        def reparameterize(self, mu, logvar):
            pass
            
        def forward(self, x):
            pass
            
    def vae_loss(recon, x, mu, logvar, beta=1.0):
        # TODO: Реализовать ELBO
        pass
    
    # === SUBMISSION (end) ===
    return Decoder, Encoder, VAE, vae_loss

@app.cell
def __():
    # TODO: Training loop
    # TODO: visualize_latent_space
    # TODO: interpolate_latent
    return

@app.cell
def __(Path, json):
    # === ANTI-CHEAT BLOCK ===
    # Вместо ML3_HIDDEN_TEST (как в classification задачах), мы просим
    # вас сохранить финальные метрики в json файл. Judge-агент задаст
    # вам вопросы на основе этих чисел.
    
    # Замените эти значения на ваши реальные!
    final_results = {
        "final_elbo": 0.0,
        "kl_divergence": 0.0,
        "reconstruction_loss": 0.0,
        "num_epochs": 0
    }
    
    Path("submission").mkdir(exist_ok=True)
    with open("submission/results.json", "w") as f:
        json.dump(final_results, f)
        
    print("Saved results.json")
    return final_results,

if __name__ == "__main__":
    app.run()

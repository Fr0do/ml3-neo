"""Seminar Generative — Diffusion (advanced level)."""

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
        # Семинар Generative — Диффузия (advanced)

        ## Задача

        1. Реализовать DDPM на MNIST (линейный schedule, 200 шагов).
        2. Показать forward noising (sequence of images).
        3. Обучить маленький U-Net.
        4. Сравнить sampling quality при 200 vs 50 vs 20 шагах (DDIM approximation).

        ## Чек-лист реализации

        - [ ] Forward noising
        - [ ] U-Net
        - [ ] Training loop
        - [ ] DDIM sampling
        """
    )
    return


@app.cell
def __(torch, nn):
    # === SUBMISSION (start) ===
    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            # TODO: implement
            
        def forward(self, x, t):
            # TODO: implement
            return x
            
    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
        
    # TODO: implement forward noising and training loop
    # === SUBMISSION (end) ===
    return SimpleUNet, linear_beta_schedule


if __name__ == "__main__":
    app.run()

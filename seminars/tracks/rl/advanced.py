"""Seminar RL — Reinforcement Learning (advanced level)."""

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
        # Семинар RL — Reinforcement Learning (advanced)

        ## Задача

        1. Реализовать REINFORCE и A2C на CartPole-v1 (gymnasium).
        2. Сравнить кривые обучения (reward vs episodes).
        3. Использовать `mo.ui.slider` для $\gamma$ — показать влияние discount factor.
        4. (Опционально) PPO.

        ## Чек-лист реализации

        - [ ] REINFORCE agent
        - [ ] A2C agent
        - [ ] Training loop for CartPole
        - [ ] Slider for $\gamma$ ablation
        """
    )
    return


@app.cell
def __(mo):
    gamma_slider = mo.ui.slider(start=0.5, stop=0.99, step=0.01, value=0.99, label="Gamma (discount)")
    return gamma_slider,


@app.cell
def __(gamma_slider):
    gamma_slider
    return


@app.cell
def __(torch, nn, gamma_slider):
    # === SUBMISSION (start) ===
    class PolicyNet(nn.Module):
        def __init__(self, obs_dim, act_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.ReLU(),
                nn.Linear(64, act_dim)
            )
            
        def forward(self, x):
            return self.net(x)

    def train_reinforce(gamma=gamma_slider.value):
        # TODO: training loop
        pass
    # === SUBMISSION (end) ===
    return PolicyNet, train_reinforce


if __name__ == "__main__":
    app.run()

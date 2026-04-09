"""Seminar 08 — Generative Models (advanced level).

Реактивный Marimo notebook. Запуск:
    marimo edit seminars/08-generative/advanced.py

Цель семинара (advanced):
    1. Реализовать упрощённый DDPM (Denoising Diffusion Probabilistic Model) с нуля.
    2. Реализовать Forward process (зашумление).
    3. Обучить Reverse process (денойзинг) на MNIST.
    4. Сгенерировать новые сэмплы.
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
        # Семинар 08 — DDPM (advanced)

        Здесь мы пишем диффузию с нуля. Мы будем использовать 50 шагов диффузии
        с линейным расписанием (linear schedule).
        В качестве базовой модели возьмем упрощенную сверточную сеть, обусловленную
        на шаг времени $t$.
        """
    )
    return

@app.cell
def __():
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    return DataLoader, nn, np, plt, torch, torchvision, transforms

@app.cell
def __(torch):
    # Параметры диффузии
    T = 50
    beta_start = 1e-4
    beta_end = 0.02
    
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return T, alphas, alphas_cumprod, beta_end, beta_start, betas

@app.cell
def __(alphas_cumprod, torch):
    def q_sample(x_start, t, noise=None):
        """Forward process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return q_sample,

@app.cell
def __(DataLoader, q_sample, torchvision, transforms):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # В диффузии данные от -1 до 1
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return dataset, transform

@app.cell
def __(dataset, plt, q_sample, torch):
    import marimo as mo
    
    # Визуализация forward process
    img, _ = dataset[0]
    img = img.unsqueeze(0)
    
    timesteps = [0, 10, 25, 49]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t])
        x_t = q_sample(img, t_tensor)
        ax = axes[i]
        ax.imshow(x_t.squeeze().numpy(), cmap='gray')
        ax.set_title(f't = {t}')
        ax.axis('off')
        
    mo.ui.pyplot(fig)
    return axes, fig, img, timesteps

@app.cell
def __(nn, torch):
    class SimpleUNet(nn.Module):
        """Очень упрощенная U-Net-подобная сеть без skip connections для скорости."""
        def __init__(self):
            super().__init__()
            self.time_embed = nn.Embedding(50, 32)
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 1, 3, padding=1)
            
        def forward(self, x, t):
            t_emb = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)
            h = torch.relu(self.conv1(x))
            h = h + t_emb
            h = torch.relu(self.conv2(h))
            out = self.conv3(h)
            return out
            
    return SimpleUNet,

@app.cell
def __(DataLoader, SimpleUNet, T, dataset, mo, q_sample, torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = SimpleUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    epochs = 3 # Для семинара хватит
    mo.md("Training Reverse Process...")
    
    model.train()
    for epoch in range(epochs):
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            t = torch.randint(0, T, (images.size(0),), device=device).long()
            noise = torch.randn_like(images).to(device)
            
            x_t = q_sample(images, t.cpu(), noise.cpu()).to(device)
            
            noise_pred = model(x_t, t)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
    return dataloader, device, epochs, model, optimizer

@app.cell
def __(T, alphas, alphas_cumprod, betas, device, model, plt, torch):
    import marimo as mo
    
    @torch.no_grad()
    def p_sample(model, x, t, t_index):
        betas_t = betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t_index])
        sqrt_recip_alphas_t = torch.sqrt(1.0 / alphas[t_index])
        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = betas_t # упрощенно
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
            
    # Генерация
    model.eval()
    img = torch.randn((1, 1, 28, 28)).to(device)
    
    for i in reversed(range(0, T)):
        t_tensor = torch.full((1,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t_tensor, i)
        
    fig2, ax2 = plt.subplots()
    ax2.imshow(img.cpu().squeeze().numpy(), cmap='gray')
    ax2.axis('off')
    ax2.set_title("Generated Sample")
    mo.ui.pyplot(fig2)
    return ax2, fig2, img, p_sample

if __name__ == "__main__":
    app.run()

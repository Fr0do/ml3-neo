"""Seminar 08 — Generative Models (basic level).

Реактивный Marimo notebook. Запуск:
    marimo edit seminars/08-generative/basic.py

Цель семинара (basic):
    1. Реализовать VAE на датасете MNIST.
    2. Понять Reparameterization trick.
    3. Исследовать латентное пространство (визуализация, бета-VAE).
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
        # Семинар 08 — VAE (basic)

        Перед запуском убедитесь, что прочитали [лекцию](../../lectures/08-generative/lecture.qmd).

        План:
        1. Собрать `Encoder` ($\mu$, $\sigma$) и `Decoder`.
        2. Реализовать `reparameterization trick` и функцию потерь (ELBO).
        3. Обучить модель на MNIST (~5 эпох).
        4. Поиграть с $eta$-VAE через слайдер и визуализировать 2D латентное пространство.
        """
    )
    return

@app.cell
def __():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    return DataLoader, F, nn, np, plt, torch, torchvision, transforms

@app.cell
def __(mo):
    batch_size = mo.ui.slider(32, 256, value=128, step=32, label="Batch size")
    beta = mo.ui.slider(0.0, 5.0, value=1.0, step=0.1, label="Beta (KL weight)")
    epochs = mo.ui.slider(1, 10, value=5, step=1, label="Epochs")
    mo.hstack([batch_size, beta, epochs])
    return batch_size, beta, epochs

@app.cell
def __(DataLoader, batch_size, torchvision, transforms):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size.value, shuffle=True)
    return dataloader, dataset, transform

@app.cell
def __(nn, torch):
    class VAE(nn.Module):
        def __init__(self, latent_dim=2):
            super().__init__()
            # Encoder
            self.enc_conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
            self.enc_conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
            self.fc_mu = nn.Linear(32 * 7 * 7, latent_dim)
            self.fc_logvar = nn.Linear(32 * 7 * 7, latent_dim)
            
            # Decoder
            self.dec_fc = nn.Linear(latent_dim, 32 * 7 * 7)
            self.dec_conv1 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
            self.dec_conv2 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)

        def encode(self, x):
            h = torch.relu(self.enc_conv1(x))
            h = torch.relu(self.enc_conv2(h))
            h = h.view(h.size(0), -1)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            h = torch.relu(self.dec_fc(z))
            h = h.view(h.size(0), 32, 7, 7)
            h = torch.relu(self.dec_conv1(h))
            return torch.sigmoid(self.dec_conv2(h))

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            return x_recon, mu, logvar

    def loss_function(recon_x, x, mu, logvar, beta_val):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta_val * KLD

    return VAE, loss_function

@app.cell
def __(VAE, beta, dataloader, epochs, loss_function, mo, torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = VAE(latent_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    mo.md("Training VAE... (This may take a minute)")
    
    model.train()
    for epoch in range(epochs.value):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, beta.value)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {train_loss / len(dataloader.dataset):.4f}')
        
    mo.md("Training complete!")
    return device, model, optimizer

@app.cell
def __(dataset, device, model, plt, torch):
    @torch.no_grad()
    def plot_latent_space(model, dataset, num_samples=1000):
        model.eval()
        loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
        data, labels = next(iter(loader))
        data = data.to(device)
        mu, _ = model.encode(data)
        mu = mu.cpu().numpy()
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(mu[:, 0], mu[:, 1], c=labels.numpy(), cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('2D Latent Space (MNIST)')
        plt.xlabel('z1')
        plt.ylabel('z2')
        return plt.gcf()
        
    fig = plot_latent_space(model, dataset)
    return fig, plot_latent_space

@app.cell
def __(fig, mo):
    mo.ui.pyplot(fig)
    return

if __name__ == "__main__":
    app.run()

#!/usr/bin/env python3
"""PyTorch DataLoader example: train a simple autoencoder on .1pz data.

Requires: pip install singlet[torch]

This demonstrates zero-copy sparse tensor loading from .1pz files
directly into PyTorch's sparse CSR format for ML training.
"""

import torch
import torch.nn as nn
from singlet.torch import DataLoader, OnePZDataset

# Load dataset — each item is one cell (sparse vector of gene counts)
dataset = OnePZDataset("counts.1pz", normalize=True)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

print(f"Dataset: {len(dataset)} cells × {dataset.n_genes} genes")


# Simple autoencoder
class Autoencoder(nn.Module):
    def __init__(self, n_genes, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_genes),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


model = Autoencoder(dataset.n_genes).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(10):
    total_loss = 0
    for batch in loader:
        # batch is a sparse CSR tensor — convert to dense for this simple example
        x = batch.to_dense().cuda()

        recon, latent = model(x)
        loss = loss_fn(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: loss = {total_loss / len(loader):.4f}")

print("\nTraining complete. Latent dim:", model.encoder[-1].out_features)

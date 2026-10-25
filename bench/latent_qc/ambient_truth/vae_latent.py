"""Per-sample scVI-style VAE on the called cells -> latent means saved to $DATA/LIB/vae_VIEW.npy.  Run with `rtt python -I` (GPU).
Encoder: log1p(CP10k) of 2000 HVGs -> 128 -> 128 -> (mu, logvar) in R^10; decoder: z -> 128 -> 128 -> softmax over the HVGs,
likelihood = multinomial on the HVG counts (library size absorbed), KL weight 1. 150 epochs, Adam 1e-3, batch 256, seed 0.
Usage: vae_latent.py LIB VIEW"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlib  # noqa: E402

lib, view = sys.argv[1:3]
ds = atlib.load(lib, view)
torch.manual_seed(0); np.random.seed(0)
X = ds.C.T.tocsr().astype(np.float32)
N = np.asarray(X.sum(1)).ravel()
L = X.multiply(1e4 / np.maximum(N, 1)[:, None]).tocsr(); L.data = np.log1p(L.data)
mean = np.asarray(L.mean(0)).ravel(); var = np.asarray(L.multiply(L).mean(0)).ravel() - mean ** 2
hv = np.argsort(-var)[:2000]
dev = "cuda" if torch.cuda.is_available() else "cpu"
Y = torch.tensor(X[:, hv].toarray(), device=dev)
Lg = torch.tensor(L[:, hv].toarray(), device=dev)
G = Y.shape[1]


class VAE(nn.Module):
    def __init__(s, d=10, h=128):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(G, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        s.mu, s.lv = nn.Linear(h, d), nn.Linear(h, d)
        s.dec = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, G))

    def forward(s, x):
        e = s.enc(x); mu, lv = s.mu(e), s.lv(e).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return torch.log_softmax(s.dec(z), 1), mu, lv


m = VAE().to(dev)
opt = torch.optim.Adam(m.parameters(), 1e-3)
n = Y.shape[0]
for ep in range(150):
    perm = torch.randperm(n, device=dev)
    tot = 0.0
    for i in range(0, n, 256):
        b = perm[i:i + 256]
        lp, mu, lv = m(Lg[b])
        nll = -(Y[b] * lp).sum(1)
        kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(1)
        loss = (nll + kl).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(b)
    if ep % 30 == 0:
        print(lib, view, "epoch", ep, "loss", tot / n, flush=True)
with torch.no_grad():
    Z = m.mu(m.enc(Lg)).cpu().numpy()
np.save(f"{atlib.DATA}/{lib}/vae_{view}.npy", Z)
print("saved", Z.shape)

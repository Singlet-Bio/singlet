"""Latent-space doublet scores (GPU).  Usage: latent_gpu.py NAME [solo] [atlas]

solo   Solo-style (Bernstein et al. 2020): an scVI-style VAE (negative binomial decoder, 10-d latent, one 128-unit layer) trained
       on the sample's 2,000 HVG counts; 2n simulated doublets (summed raw counts of random pairs) are encoded and a classifier
       on the latent separates them from the real cells; score = classifier probability on the real cells.
atlas  the cell type network trained on 2.8M CELLxGENE cells (singlet.celltype_nn, model m1_notissue) as a fixed, atlas-learned
       encoder: its 2,048-d hidden layer embeds real cells and 2n simulated doublets.
         atlas_knn    fraction of simulated doublets among each real cell's k nearest neighbours (PCA-50 of the embedding)
         atlas_clf    5-fold cross-fitted classifier, simulated vs real, on the same PCA-50 (out-of-fold score per real cell)
         ct_entropy   entropy of the network's calibrated cell type distribution; ct_maxprob = 1 - its largest probability
         atlas_nnls   two cell types in one droplet: best non-negative fit of the cell's vocabulary profile by two atlas
                      cell-type centroids vs one (relative residual drop; candidates = the 8 best single centroids)
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from dbl_common import LQC, Timer, has_score, hvg, load, put_score, simulate_doublets

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "/mnt/projects/debruinz_project/debruinz/singlet/data/celltype_nn/models/m1_notissue/model.pt"


# ---------------------------------------------------------------- Solo-style VAE
class VAE(nn.Module):
    def __init__(self, g, h=128, z=10):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(g, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1))
        self.mu, self.lv = nn.Linear(h, z), nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, g))
        self.log_theta = nn.Parameter(torch.zeros(g))

    def encode(self, x):
        e = self.enc(torch.log1p(x))
        return self.mu(e), self.lv(e).clamp(-10, 10)

    def loss(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        lib = x.sum(1, keepdim=True)
        m = F.softmax(self.dec(z), 1) * lib
        th = self.log_theta.exp()
        ll = (torch.lgamma(x + th) - torch.lgamma(th) - torch.lgamma(x + 1) + th * (torch.log(th) - torch.log(th + m + 1e-8))
              + x * (torch.log(m + 1e-8) - torch.log(th + m + 1e-8))).sum(1)
        kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)
        return (-ll + kl).mean()


def dense(X, rows=None):
    A = X if rows is None else X[rows]
    return torch.from_numpy(np.asarray(A.todense(), np.float32)).to(DEV)


def train_vae(X, seed=0):
    torch.manual_seed(seed)
    n = X.shape[0]
    epochs = int(min(400, max(40, round(20000 / n * 400))))
    m = VAE(X.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xt = dense(X)
    bs = 256
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            if len(b) < 3:
                continue
            opt.zero_grad()
            m.loss(Xt[b]).backward()
            opt.step()
    m.eval()
    return m, epochs


def fit_classifier(Z, y, seed=0, epochs=100, hidden=128, wd=1e-4):
    """MLP real (0) vs simulated (1) with a 20% validation split and early stopping; returns the model."""
    torch.manual_seed(seed)
    n = len(y)
    perm = torch.randperm(n, device=DEV)
    va, tr = perm[: n // 5], perm[n // 5:]
    clf = nn.Sequential(nn.Linear(Z.shape[1], hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, 1)).to(DEV)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=wd)
    best, state, bad = 1e9, None, 0
    for ep in range(epochs):
        clf.train()
        p = tr[torch.randperm(len(tr), device=DEV)]
        for s in range(0, len(p), 512):
            b = p[s:s + 512]
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(clf(Z[b]).squeeze(1), y[b]).backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            v = F.binary_cross_entropy_with_logits(clf(Z[va]).squeeze(1), y[va]).item()
        if v < best - 1e-4:
            best, state, bad = v, {k: t.clone() for k, t in clf.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10:
                break
    clf.load_state_dict(state)
    clf.eval()
    return clf


def solo(d, rng):
    with Timer() as t:
        X = d["X"]
        h = hvg(X, 2000)
        Xh = X[:, h].tocsr().astype(np.float32)
        vae, epochs = train_vae(Xh)
        S, _ = simulate_doublets(Xh, 2 * X.shape[0], rng)
        with torch.no_grad():
            zr = torch.cat([vae.encode(dense(Xh[s:s + 8192]))[0] for s in range(0, Xh.shape[0], 8192)])
            zs = torch.cat([vae.encode(dense(S[s:s + 8192]))[0] for s in range(0, S.shape[0], 8192)])
        Z = torch.cat([zr, zs])
        y = torch.cat([torch.zeros(len(zr)), torch.ones(len(zs))]).to(DEV)
        clf = fit_classifier(Z, y)
        with torch.no_grad():
            sc = torch.sigmoid(clf(zr)).squeeze(1).cpu().numpy()
    put_score(d["name"], "solo", sc, t.s, vae_epochs=epochs, device=str(DEV))


# ---------------------------------------------------------------- atlas-trained encoder
def vocab_projection(pred, genes, species):
    """genes x vocab 0/1 matrix: Ensembl ids (optionally prefixed hg19_/mm10_/GRCh38_) through the network's vocabulary (mouse
    by one-to-one orthologs); gene symbols by the vocabulary's human symbols (upper case, so most mouse symbols map too)."""
    v = pred.vocab
    g = np.array([str(x) for x in genes])
    strip = np.array([x.split("_", 1)[1] if x[:5] in ("hg19_", "mm10_", "GRCh3", "GRCm3", "hg38_") and "_" in x else x for x in g])
    strip = np.array([x.split(".")[0] for x in strip])
    cols = np.full(len(g), -1)
    is_ens = np.char.startswith(strip, "ENS")
    if is_ens.mean() > 0.5:
        hum = np.char.startswith(strip, "ENSG")
        cols[hum] = v.columns(strip[hum], "human")
        mou = np.char.startswith(strip, "ENSMUSG")
        cols[mou] = v.columns(strip[mou], "mouse")
    else:
        sym = {s.upper(): i for i, s in enumerate(v.symbols)}
        cols = np.array([sym.get(s.upper(), -1) for s in strip])
    keep = np.flatnonzero(cols >= 0)
    P = sp.csr_matrix((np.ones(len(keep), np.float32), (keep, cols[keep])), shape=(len(g), len(v)))
    return P, len(keep)


def knn_fraction(E, n_real, k):
    """For each of the first n_real rows of E: fraction of its k nearest neighbours (self excluded) that are simulated."""
    out = torch.zeros(n_real, device=DEV)
    for s in range(0, n_real, 4096):
        dd = torch.cdist(E[s:min(s + 4096, n_real)], E)
        dd[torch.arange(dd.shape[0]), torch.arange(s, s + dd.shape[0])] = float("inf")
        nb = dd.topk(k, largest=False).indices
        out[s:s + dd.shape[0]] = (nb >= n_real).float().mean(1)
    return out.cpu().numpy()


def atlas(d, rng):
    from singlet.celltype_nn import CellTypePredictor
    from singlet.celltype_nn.features import transform

    pred = CellTypePredictor.load(MODEL)
    net = pred.net.to(DEV)
    cen = np.load(f"{LQC}/dbl/atlas_centroids.npz", allow_pickle=True)
    with Timer() as t0:
        Pm, n_mapped = vocab_projection(pred, d["genes"], d["species"])
        V = (d["X"].astype(np.float32) @ Pm).tocsr()
        n = V.shape[0]
        S, _ = simulate_doublets(V, 2 * n, rng)
        A = sp.vstack([V, S]).tocsr()

        emb, prob = [], []
        with torch.no_grad():
            for s in range(0, A.shape[0], 8192):
                e = net.embed(transform(dense(A, np.arange(s, min(s + 8192, A.shape[0]))), net.transform))
                emb.append(e)
                prob.append(F.softmax(net.head(e) / net.temperature, 1))
        E = torch.cat(emb)
        Pr = torch.cat(prob)[:n]
        del prob
        E = E - E.mean(0)
        U, Sg, Vt = torch.pca_lowrank(E, q=50, center=False)
        Z = E @ Vt
    with Timer() as t1:
        k = max(10, int(round(0.5 * np.sqrt(n))))
        knn = knn_fraction(Z, n, k)
    put_score(d["name"], "atlas_knn", knn, t0.s + t1.s, k=k, genes_mapped=n_mapped, device=str(DEV))
    with Timer() as t2:
        y = torch.cat([torch.zeros(n), torch.ones(2 * n)]).to(DEV)
        Zs = (Z - Z.mean(0)) / Z.std(0).clamp_min(1e-6)
        fold = torch.from_numpy(rng.integers(0, 5, n)).to(DEV)
        fold_sim = torch.from_numpy(rng.integers(0, 5, 2 * n)).to(DEV)
        oof = np.zeros(n)
        for f in range(5):
            tr = torch.cat([torch.nonzero(fold != f).squeeze(1), n + torch.nonzero(fold_sim != f).squeeze(1)])
            clf = fit_classifier(Zs[tr], y[tr], seed=f)
            te = torch.nonzero(fold == f).squeeze(1)
            with torch.no_grad():
                oof[te.cpu().numpy()] = torch.sigmoid(clf(Zs[te])).squeeze(1).cpu().numpy()
    put_score(d["name"], "atlas_clf", oof, t0.s + t2.s)
    P = Pr.cpu().numpy().astype(np.float64)
    put_score(d["name"], "ct_entropy", -(P * np.log(P + 1e-12)).sum(1), t0.s)
    put_score(d["name"], "ct_maxprob", 1 - P.max(1), t0.s)
    # two atlas cell types in one droplet
    with Timer() as t3:
        M = torch.from_numpy(cen["M"]).to(DEV).double()
        Vd = V.astype(np.float64)
        tot = np.asarray(Vd.sum(1)).ravel()
        mean_p = torch.from_numpy(np.asarray((sp.diags(1 / np.maximum(tot, 1)) @ Vd).mean(0)).ravel()).to(DEV)
        w = 1 / torch.sqrt(mean_p + 1e-6)
        Mw = M * w
        mm = (Mw * Mw).sum(1)
        rel = np.zeros(n)
        minor = np.zeros(n)
        for s in range(0, n, 2048):
            r = slice(s, min(n, s + 2048))
            p = torch.from_numpy(np.asarray((sp.diags(1 / np.maximum(tot[r], 1)) @ Vd[r]).todense())).to(DEV) * w
            pp = (p * p).sum(1)
            G = p @ Mw.T
            fit1 = torch.where(G > 0, G * G / mm, torch.zeros_like(G))
            top = fit1.topk(8, 1).indices
            sse1 = pp - fit1.gather(1, top[:, :1]).squeeze(1)
            best2 = sse1.clone()
            mn = torch.zeros_like(best2)
            for a in range(8):
                for b in range(a + 1, 8):
                    ka, kb = top[:, a], top[:, b]
                    caa, cbb = mm[ka], mm[kb]
                    cab = (Mw[ka] * Mw[kb]).sum(1)
                    ga, gb = G.gather(1, ka[:, None]).squeeze(1), G.gather(1, kb[:, None]).squeeze(1)
                    det = (caa * cbb - cab ** 2).clamp_min(1e-12)
                    wa = (cbb * ga - cab * gb) / det
                    wb = (caa * gb - cab * ga) / det
                    sse = pp - (wa * ga + wb * gb)
                    upd = (wa > 0) & (wb > 0) & (sse < best2)
                    best2 = torch.where(upd, sse, best2)
                    mn = torch.where(upd, torch.minimum(wa, wb) / (wa + wb), mn)
            rel[r] = (1 - best2 / sse1.clamp_min(1e-12)).cpu().numpy()
            minor[r] = mn.cpu().numpy()
    put_score(d["name"], "atlas_nnls", rel, t3.s, n_centroids=int(M.shape[0]))
    put_score(d["name"], "atlas_nnls_minor", minor, 0.0)
    np.save(f"{LQC}/dbl/scores/{d['name']}/ct_argmax.npy", np.array(pred.space.terms, dtype=object)[P.argmax(1)].astype(str))


if __name__ == "__main__":
    name = sys.argv[1]
    d = load(name)
    for m in sys.argv[2:]:
        key = {"solo": "solo", "atlas": "atlas_nnls"}[m]
        if has_score(name, key) and not os.environ.get("FORCE"):
            continue
        globals()[m](d, np.random.default_rng(0))

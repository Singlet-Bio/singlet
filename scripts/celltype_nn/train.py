#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Train the cell type network on the Census corpus (author terms + ensemble refinements) and calibrate it.

  python scripts/celltype_nn/train.py --data $D --out $D/models/<name> [--epochs 12] [--no-augment] [--no-ensemble]

Each cell's loss is the expected-credit loss against its author term (folded into the output space); a cell whose author term
the ensemble refines adds the same loss against the refinement with weight ``--ensemble-weight``. Counts are augmented on the
GPU (:class:`singlet.celltype_nn.features.Augmenter`). The temperature is fitted on the validation collections, half of them
augmented, so that it holds under the shift the augmentation stands for. Writes ``model.pt`` (weights, vocabulary, terms,
temperature) and ``train.json`` (history, validation metrics).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


class Batches:
    """Shuffled minibatches of CSR rows as dense float32 tensors (built in DataLoader workers)."""

    def __init__(self, X: sps.csr_matrix, idx: np.ndarray, batch: int, seed: int, shuffle: bool = True):
        self.X, self.idx, self.batch, self.seed, self.shuffle = X, idx, batch, seed, shuffle
        self.epoch = 0

    def order(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        o = rng.permutation(self.idx) if self.shuffle else self.idx
        return [o[i : i + self.batch] for i in range(0, len(o), self.batch)]

    def __call__(self, rows):
        import torch

        rows = np.asarray(rows[0])
        return torch.from_numpy(self.X[rows].toarray()), torch.from_numpy(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--hidden", default="1024,512")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--input-dropout", type=float, default=0.1)
    ap.add_argument("--transform", default="log", choices=["log", "logz", "rank"])
    ap.add_argument("--tissue-context", action="store_true", help="feed the guide's tissue groups of Census tissue_general")
    ap.add_argument("--context-dropout", type=float, default=0.3, help="fraction of training cells shown without context")
    ap.add_argument("--ensemble-weight", type=float, default=0.5)
    ap.add_argument("--coarse-credit", type=float, default=0.25, help="beta: credit of an ancestor of the target = beta * IC ratio")
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--singlet-pseudo", type=Path, default=None, help="directory written by singlet_pseudo.py run")
    ap.add_argument("--pseudo-weight", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import pandas as pd
    import torch

    from singlet.celltype_nn import model as M
    from singlet.celllabels.guide import TISSUE_GROUPS
    from singlet.celltype_nn.features import AugmentConfig, Augmenter, GeneVocab, tissue_context, transform
    from singlet.celltype_nn.ontology import TermSpace

    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    X = sps.load_npz(a.data / "corpus_counts.npz").tocsr().astype(np.float32)
    obs = pd.read_parquet(a.data / "corpus_obs.parquet")
    cells = pd.read_parquet(a.data / "space_cells.parquet")
    assert (cells["cell_id"].to_numpy() == obs["cell_id"].to_numpy()).all()
    terms = json.loads((a.data / "terms.json").read_text())
    vocab = GeneVocab.from_dict(json.loads((a.data / "vocab.json").read_text()))
    sp = TermSpace(terms["terms"])
    author = cells["author_term"].to_numpy(object)
    refine = np.full(len(obs), None, object)
    tp = a.data / "targets_ensemble.parquet"
    if tp.exists() and not a.no_ensemble:
        t = pd.read_parquet(tp)
        pos = pd.Series(np.arange(len(obs)), index=obs["cell_id"].to_numpy())
        refine[pos.loc[t["cell_id"].to_numpy()].to_numpy()] = t["refine_term"].to_numpy(object)
    ctx_text = obs["tissue_general"].to_numpy(object)
    split = obs["split"].to_numpy(object)
    weight = np.ones(len(obs), np.float32)
    n_pseudo = 0
    if a.singlet_pseudo is not None:
        # Singlet cells labelled by the ensemble consensus: extra training rows with their GEO tissue text as context
        from singlet_pseudo import load_pseudo

        Xp, op = load_pseudo(a.singlet_pseudo)
        if Xp is not None:
            n_pseudo = Xp.shape[0]
            X = sps.vstack([X, Xp]).tocsr()
            author = np.concatenate([author, op["consensus"].to_numpy(object)])
            refine = np.concatenate([refine, np.full(n_pseudo, None, object)])
            ctx_text = np.concatenate([ctx_text, op["tissue"].to_numpy(object)])
            split = np.concatenate([split, np.full(n_pseudo, "train", object)])
            weight = np.concatenate([weight, np.full(n_pseudo, a.pseudo_weight, np.float32)])
    ok = pd.notna(author)
    # target ids: author and refinement terms index one credit table
    tlist = sorted(set(author[ok].tolist()) | {r for r in refine if isinstance(r, str)})
    tix = {x: i for i, x in enumerate(tlist)}
    credit = torch.from_numpy(sp.credit_matrix(tlist, a.coarse_credit)).to(dev)
    closure = (credit >= 0.999).float()
    # for the roll-up call metric: closure term a is consistent with target t when a is t, an ancestor or a descendant of it
    Rup = torch.from_numpy(sp.R).to(dev)
    key_c = torch.from_numpy(sp.ic_closure + 1e-6 * sp.depth_closure).to(dev)
    cons = np.zeros((len(tlist), len(sp.closure)), np.float32)
    for i, t in enumerate(tlist):
        at = sp.ancestors(t)
        for j, c in enumerate(sp.closure):
            if c in at or t in sp.ancestors(c):
                cons[i, j] = 1
    cons = torch.from_numpy(cons).to(dev)
    a_idx = torch.tensor([tix.get(x, -1) if isinstance(x, str) else -1 for x in author], device=dev)
    r_idx = torch.tensor([tix.get(x, -1) if isinstance(x, str) else -1 for x in refine], device=dev)
    tr = np.flatnonzero(ok & (split == "train"))
    va = np.flatnonzero(ok & (split == "val"))
    print(f"{len(tr)} train / {len(va)} val cells, {len(sp)} terms, {len(vocab)} genes, "
          f"{int(pd.notna(refine[tr]).sum())} refined, {n_pseudo} Singlet pseudo-labelled, device {dev}, {time.time() - t0:.0f}s",
          flush=True)
    W = torch.from_numpy(weight).to(dev)

    groups = [g for g in TISSUE_GROUPS if g not in ("tumor", "embryo")] if a.tissue_context else []
    CTX = torch.from_numpy(tissue_context(ctx_text, groups)).to(dev) if groups else None
    ir = np.load(a.data / "intron_ratio.npy") if (a.data / "intron_ratio.npy").exists() else None
    aug = None if a.no_augment else Augmenter(len(vocab), ir, AugmentConfig(), dev, seed=a.seed)
    hidden = tuple(int(h) for h in a.hidden.split(","))
    net = M.CellTypeNet(len(vocab), len(sp), hidden=hidden, dropout=a.dropout, input_dropout=a.input_dropout,
                        transform=a.transform, n_context=len(groups)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    train_b = Batches(X, tr, a.batch, a.seed)
    steps = a.epochs * len(train_b.order())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)

    def loader(b: Batches):
        return torch.utils.data.DataLoader(
            b.order(), batch_size=1, shuffle=False, num_workers=a.workers, collate_fn=b, persistent_workers=False,
            prefetch_factor=4 if a.workers else None,
        )

    def evaluate(idx, augment: bool):
        net.eval()
        logits, rows = [], []
        b = Batches(X, idx, 4096, 0, shuffle=False)
        with torch.no_grad():
            for c, r in loader(b):
                c = c.to(dev, non_blocking=True)
                if augment and aug is not None:
                    c = aug(c)
                logits.append(net(transform(c, a.transform), None if CTX is None else CTX[r.to(dev)]))
                rows.append(r)
        net.train()
        return torch.cat(logits), torch.cat(rows).to(dev)

    def metrics(logits, rows, T=1.0):
        p = torch.softmax(logits / T, 1)
        cr = credit[a_idx[rows]]
        am = p.argmax(1)
        c_am = cr.gather(1, am[:, None]).ravel()
        P = p @ Rup
        k = torch.where(P >= 0.5, key_c[None, :] + 1e-3 * P, torch.full_like(P, -1e9)).argmax(1)
        call_ok = cons[a_idx[rows]].gather(1, k[:, None]).ravel()
        depth_call = torch.from_numpy(sp.depth_closure).to(dev)[k]
        return dict(
            call_consistent=float(call_ok.mean()),
            call_mean_depth=float(depth_call.mean()),
            loss=float(M.credit_loss(logits / T, cr)),
            closure_nll=float(M.credit_loss(logits / T, closure[a_idx[rows]])),
            expected_credit=float((p * cr).sum(1).mean()),
            argmax_full_credit=float((c_am >= 0.999).float().mean()),
            argmax_wrong=float((c_am <= 0).float().mean()),
        )

    hist, best = [], None
    for ep in range(a.epochs):
        train_b.epoch = ep
        tl, nb, te = 0.0, 0, time.time()
        for c, rows in loader(train_b):
            c = c.to(dev, non_blocking=True)
            rows = rows.to(dev)
            if aug is not None:
                c = aug(c)
            cx = None
            if CTX is not None:
                cx = CTX[rows] * (torch.rand(len(rows), 1, device=dev) >= a.context_dropout).float()
            logits = net(transform(c, a.transform), cx)
            loss = M.credit_loss(logits, credit[a_idx[rows]], W[rows])
            rr = r_idx[rows]
            has = rr >= 0
            if has.any():
                loss = loss + a.ensemble_weight * M.credit_loss(logits[has], credit[rr[has]]) * has.float().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            tl += float(loss)
            nb += 1
        lv, rv = evaluate(va, augment=False)
        la, ra = evaluate(va, augment=True)
        h = dict(epoch=ep, train_loss=tl / max(nb, 1), val=metrics(lv, rv), val_aug=metrics(la, ra), seconds=time.time() - te)
        hist.append(h)
        print(json.dumps(h), flush=True)
        # keep the epoch with the best validation closure likelihood (held-out collections, augmented and not)
        score = h["val"]["closure_nll"] + h["val_aug"]["closure_nll"]
        if best is None or score < best[0]:
            best = (score, ep, {k: v.detach().clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[2])
    print(f"best epoch {best[1]}", flush=True)

    # calibration: one temperature on validation logits, half clean and half augmented. The partial credit of coarser terms
    # is not a proper scoring rule (it rewards sharpening), so the temperature is fitted on the closure likelihood
    # -log P(author term or a descendant), which is proper for labels of any depth.
    lv, rv = evaluate(va, augment=False)
    la, ra = evaluate(va, augment=True)
    L = torch.cat([lv, la])
    R = torch.cat([rv, ra])
    T = M.fit_temperature(L, closure[a_idx[R]])
    net.temperature.fill_(T)
    final = dict(temperature=T, best_epoch=best[1], val=metrics(lv, rv, T), val_aug=metrics(la, ra, T))
    print("calibrated", json.dumps(final), flush=True)
    M.save(
        net.cpu(),
        a.out / "model.pt",
        dict(
            vocab=vocab.to_dict(),
            terms=sp.terms,
            tissue_groups=groups,
            cl_version=sp.o.version,
            census=json.loads((a.data / "plan.json").read_text()).get("census_version"),
            args=vars(a) | {"data": str(a.data), "out": str(a.out)},
        ),
    )
    (a.out / "train.json").write_text(json.dumps(dict(history=hist, final=final, args=vars(a)), indent=1, default=str))
    print(f"done {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

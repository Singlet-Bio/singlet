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
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--hidden", default="1024,512")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--ensemble-weight", type=float, default=0.5)
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import pandas as pd
    import torch

    from singlet.celltype_nn import model as M
    from singlet.celltype_nn.features import AugmentConfig, Augmenter, GeneVocab, transform
    from singlet.celltype_nn.ontology import TermSpace

    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    X = sps.load_npz(a.data / "corpus_counts.npz").tocsr().astype(np.float32)
    obs = pd.read_parquet(a.data / "corpus_obs.parquet")
    cells = pd.read_parquet(a.data / "space_cells.parquet")
    assert (cells["soma_joinid"].to_numpy() == obs["soma_joinid"].to_numpy()).all()
    terms = json.loads((a.data / "terms.json").read_text())
    vocab = GeneVocab.from_dict(json.loads((a.data / "vocab.json").read_text()))
    sp = TermSpace(terms["terms"])
    author = cells["author_term"].to_numpy(object)
    refine = np.full(len(obs), None, object)
    tp = a.data / "targets_ensemble.parquet"
    if tp.exists() and not a.no_ensemble:
        t = pd.read_parquet(tp)
        pos = pd.Series(np.arange(len(obs)), index=obs["soma_joinid"].to_numpy())
        refine[pos.loc[t["soma_joinid"].to_numpy()].to_numpy()] = t["refine_term"].to_numpy(object)
    ok = pd.notna(author)
    # target ids: author and refinement terms index one credit table
    tlist = sorted(set(author[ok].tolist()) | {r for r in refine if isinstance(r, str)})
    tix = {x: i for i, x in enumerate(tlist)}
    credit = torch.from_numpy(sp.credit_matrix(tlist)).to(dev)
    a_idx = torch.tensor([tix.get(x, -1) if isinstance(x, str) else -1 for x in author], device=dev)
    r_idx = torch.tensor([tix.get(x, -1) if isinstance(x, str) else -1 for x in refine], device=dev)
    split = obs["split"].to_numpy()
    tr = np.flatnonzero(ok & (split == "train"))
    va = np.flatnonzero(ok & (split == "val"))
    print(f"{len(tr)} train / {len(va)} val cells, {len(sp)} terms, {len(vocab)} genes, "
          f"{int(pd.notna(refine[tr]).sum())} refined, device {dev}, {time.time() - t0:.0f}s", flush=True)

    ir = np.load(a.data / "intron_ratio.npy") if (a.data / "intron_ratio.npy").exists() else None
    aug = None if a.no_augment else Augmenter(len(vocab), ir, AugmentConfig(), dev, seed=a.seed)
    hidden = tuple(int(h) for h in a.hidden.split(","))
    net = M.CellTypeNet(len(vocab), len(sp), hidden=hidden, dropout=a.dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    train_b = Batches(X, tr, a.batch, a.seed)
    steps = a.epochs * len(train_b.order())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)

    def loader(b: Batches):
        return torch.utils.data.DataLoader(
            [[r] for r in b.order()], batch_size=1, shuffle=False, num_workers=a.workers, collate_fn=b, persistent_workers=False,
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
                logits.append(net(transform(c)))
                rows.append(r)
        net.train()
        return torch.cat(logits), torch.cat(rows).to(dev)

    def metrics(logits, rows, T=1.0):
        p = torch.softmax(logits / T, 1)
        cr = credit[a_idx[rows]]
        am = p.argmax(1)
        c_am = cr.gather(1, am[:, None]).ravel()
        return dict(
            loss=float(M.credit_loss(logits / T, cr)),
            expected_credit=float((p * cr).sum(1).mean()),
            argmax_full_credit=float((c_am >= 0.999).float().mean()),
            argmax_wrong=float((c_am <= 0).float().mean()),
        )

    hist = []
    for ep in range(a.epochs):
        train_b.epoch = ep
        tl, nb, te = 0.0, 0, time.time()
        for c, rows in loader(train_b):
            c = c.to(dev, non_blocking=True)
            rows = rows.to(dev)
            if aug is not None:
                c = aug(c)
            logits = net(transform(c))
            loss = M.credit_loss(logits, credit[a_idx[rows]])
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

    # calibration: one temperature on validation logits, half clean and half augmented
    lv, rv = evaluate(va, augment=False)
    la, ra = evaluate(va, augment=True)
    L = torch.cat([lv, la])
    R = torch.cat([rv, ra])
    T = M.fit_temperature(L, credit[a_idx[R]])
    net.temperature.fill_(T)
    final = dict(temperature=T, val=metrics(lv, rv, T), val_aug=metrics(la, ra, T))
    print("calibrated", json.dumps(final), flush=True)
    M.save(
        net.cpu(),
        a.out / "model.pt",
        dict(
            vocab=vocab.to_dict(),
            terms=sp.terms,
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

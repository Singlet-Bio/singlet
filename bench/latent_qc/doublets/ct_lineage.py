"""Lineage-level cell type probabilities from the atlas cell type network, for the real-data homotypic analysis.
Usage: ct_lineage.py NAME [NAME ...]   ->  $LQC/dbl/scores/<name>/lineage_prob.npy (cells x lineages), lineages.txt

The network's probability of a lineage is the mass on the lineage term and all its descendants (TermSpace roll-up). A true
doublet whose mass sits >= 0.9 on one lineage looks like one cell type: a homotypic doublet for any expression method.
"""
import os
import sys

import numpy as np
import torch

from dbl_common import LQC, load
from latent_gpu import MODEL, dense, vocab_projection

LINEAGES = {"CL:0000084": "T cell", "CL:0000623": "natural killer cell", "CL:0000236": "B cell", "CL:0000786": "plasma cell",
            "CL:0000576": "monocyte", "CL:0000235": "macrophage", "CL:0000451": "dendritic cell", "CL:0000233": "platelet",
            "CL:0000556": "megakaryocyte", "CL:0000232": "erythrocyte", "CL:0000097": "mast cell", "CL:0000094": "granulocyte",
            "CL:0000066": "epithelial cell", "CL:0000115": "endothelial cell", "CL:0000057": "fibroblast",
            "CL:0000192": "smooth muscle cell", "CL:0000540": "neuron", "CL:0000125": "glial cell", "CL:0000034": "stem cell"}


def descendants_mask(space, root):
    """Output terms that are root or have root among their is_a ancestors."""
    o = space.o.terms
    memo = {}

    def anc(t):
        if t in memo:
            return memo[t]
        out = {t}
        for p in o.get(t, {}).get("is_a", []) or []:
            out |= anc(p)
        memo[t] = out
        return out

    return np.array([root in anc(t) for t in space.terms])


def main(names):
    from singlet.celltype_nn import CellTypePredictor
    from singlet.celltype_nn.features import transform

    pred = CellTypePredictor.load(MODEL)
    net = pred.net
    masks = np.stack([descendants_mask(pred.space, r) for r in LINEAGES]).T.astype(np.float32)  # terms x lineages
    print("terms per lineage", dict(zip(LINEAGES.values(), masks.sum(0).astype(int))), flush=True)
    for name in names:
        d = load(name)
        P, _ = vocab_projection(pred, d["genes"], d["species"])
        V = (d["X"].astype(np.float32) @ P).tocsr()
        out = []
        with torch.no_grad():
            for s in range(0, V.shape[0], 4096):
                x = transform(dense(V, np.arange(s, min(s + 4096, V.shape[0]))).cpu(), net.transform)
                out.append(net.proba(x).numpy() @ masks)
        L = np.vstack(out)
        os.makedirs(f"{LQC}/dbl/scores/{name}", exist_ok=True)
        np.save(f"{LQC}/dbl/scores/{name}/lineage_prob.npy", L)
        open(f"{LQC}/dbl/scores/{name}/lineages.txt", "w").write("\n".join(LINEAGES.values()))
        print(name, L.shape, "median max lineage prob", np.median(L.max(1)).round(3), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

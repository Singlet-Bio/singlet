# SPDX-License-Identifier: MIT
"""Post-hoc (not in the registered plan): do truth-R2-unspliced molecules mix intron-touching and exon-only R2 reads?
Singlet labels a molecule unspliced only if EVERY read touches an intron, ambiguous if reads of both kinds (umi_dedup.h label_of).
Out: mixed_check.json = for molecules of each R2 read-mix category, the long-read class distribution."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from truth_usa import enc
X = Path(sys.argv[1])
mol = np.load(X / "truth" / "molecules.npz")
mk, mg = mol["key"], mol["gene"]
parts = [np.load(f) for f in sorted((X / "truth" / "parts").glob("b_*.npz"))]
key = np.concatenate([p["key"] for p in parts]); gene = np.concatenate([p["gene"] for p in parts])
cls = np.concatenate([p["cls"] for p in parts]); del parts
pos = np.minimum(np.searchsorted(mk, key), len(mk) - 1)
keep = (mk[pos] == key) & (mg[pos] == gene)
pos, cls = pos[keep], cls[keep]
isU = (cls == 1)
nU = np.bincount(pos, weights=isU, minlength=len(mk)); nR = np.bincount(pos, minlength=len(mk))
df = pd.read_parquet(X / "truth" / "molecules.parquet", columns=["CB", "UB", "clsLR", "clsR2", "n_pb", "n_ont"])
k2 = (enc(df.CB.tolist(), 16) << 22) | enc(df.UB.tolist(), 11)
assert (k2 == mk).all()
cat = np.where(nU == 0, "noU", np.where(nU == nR, "allU", "mixedU"))
out = {"reads_per_molecule_mean": float(nR.mean())}
for c in ("noU", "mixedU", "allU"):
    m = cat == c
    out[c] = {"molecules": int(m.sum()), "LR_class": {n: int((df.clsLR.to_numpy()[m] == v).sum()) for n, v in (("S", 0), ("U", 1), ("A", 2))}}
json.dump(out, open(X / "mixed_check.json", "w"), indent=1); print(json.dumps(out, indent=1))

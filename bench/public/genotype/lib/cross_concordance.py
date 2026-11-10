"""PRIVATE. Same-donor vs different-donor genotype concordance between the donors Singlet inferred in two runs (positive
control for a genotype-based donor merge: the same person in two lanes). Prints only summary numbers.

Usage: cross_concordance.py RUNS POOL_X POOL_Y [POOL_X POOL_Y ...]
For each donor of X: the concordance with its best-matching donor of Y (the same person), and with the second best
(a different person), at sites where both have DP >= 10 and a call.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "capabilities"))
import pools as P  # noqa: E402
import genotypes as G  # noqa: E402


def load(runs, pool):
    d = P.rundir(runs, pool) / "singlet_auto" / "private" / "donor"
    out = {}
    for f in sorted(d.glob("donor*.vcf")):
        keys, gts, dps = G.read_donor_vcf(f)
        out[f.stem] = ({k: i for i, k in enumerate(keys)}, gts[0], dps[0])
    return out


runs = Path(sys.argv[1])
args = sys.argv[2:]
for X, Y in zip(args[::2], args[1::2]):
    a, b = load(runs, X), load(runs, Y)
    same, diff = [], []
    for na, (ia, ga, da) in a.items():
        row = []
        for nb, (ib, gb, db) in b.items():
            common = [(i, ib[k]) for k, i in ia.items() if k in ib]
            ii = np.array([c[0] for c in common]); jj = np.array([c[1] for c in common])
            m = (da[ii] >= 10) & (db[jj] >= 10) & (ga[ii] >= 0) & (gb[jj] >= 0)
            row.append((float((ga[ii][m] == gb[jj][m]).mean()) if m.sum() else float("nan"), int(m.sum())))
        r = sorted(row, key=lambda t: -t[0])
        same.append(r[0][0]); diff.append(r[1][0] if len(r) > 1 else float("nan"))
        print(f"{X} {na} vs {Y}: best {r[0][0]:.3f} ({r[0][1]} sites), second {r[1][0]:.3f}" if len(r) > 1 else f"{na}: best {r[0][0]:.3f}")
    print(f"== {X} vs {Y}: same-donor concordance min {np.nanmin(same):.3f} (median {np.nanmedian(same):.3f}); best different-donor max {np.nanmax(diff):.3f}")

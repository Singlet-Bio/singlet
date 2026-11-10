"""PRIVATE. Pairwise genotype concordance between the donors Singlet inferred in one run (diagnostic for the auto-K
over-split). Prints only summary numbers per pair (fraction of shared sites with identical genotype class, and the number of
shared sites), never a genotype.

Usage: pair_concordance.py RUNS POOL [POOL ...]
Sites: both donors have DP >= 10 and a genotype call. Output rows: pool, donor_a, donor_b, n_sites, concordance, n_cells_a, n_cells_b.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "capabilities"))
import pools as P  # noqa: E402
import genotypes as G  # noqa: E402
import demux_eval as DE  # noqa: E402

runs = Path(sys.argv[1])
for pool in sys.argv[2:]:
    rd = P.rundir(runs, pool) / "singlet_auto"
    lab = DE.read_singlet(rd)["label"]
    sizes = lab[~lab.isin(DE.UNASSIGNED)].value_counts()
    vc = {}
    for f in sorted((rd / "private" / "donor").glob("donor*.vcf")):
        keys, gts, dps = G.read_donor_vcf(f)
        vc[f.stem] = (np.array(keys), gts[0], dps[0])
    names = sorted(vc)
    ref_keys = vc[names[0]][0]
    same = all((vc[n][0].shape == ref_keys.shape) and (vc[n][0] == ref_keys).all() for n in names)
    rows = []
    for a, b in itertools.combinations(names, 2):
        ka, ga, da = vc[a]; kb, gb, db = vc[b]
        if same:
            m = (da >= 10) & (db >= 10) & (ga >= 0) & (gb >= 0)
            conc = float((ga[m] == gb[m]).mean()) if m.sum() else float("nan")
            n = int(m.sum())
        else:
            ia = {k: i for i, k in enumerate(ka)}
            sel = [(ia[k], j) for j, k in enumerate(kb) if k in ia]
            ii = np.array([s[0] for s in sel]); jj = np.array([s[1] for s in sel])
            m = (da[ii] >= 10) & (db[jj] >= 10) & (ga[ii] >= 0) & (gb[jj] >= 0)
            conc = float((ga[ii][m] == gb[jj][m]).mean()) if m.sum() else float("nan")
            n = int(m.sum())
        rows.append(dict(pool=pool, donor_a=a, donor_b=b, n_sites=n, concordance=round(conc, 4),
                         n_cells_a=int(sizes.get(a, 0)), n_cells_b=int(sizes.get(b, 0))))
    d = pd.DataFrame(rows).sort_values("concordance", ascending=False)
    print(f"== {pool}: {len(names)} donors; most similar pairs:")
    print(d.head(4).to_string(index=False))
    print(f"   least similar pair concordance {d['concordance'].min()}, median {d['concordance'].median()}")

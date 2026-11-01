# SPDX-License-Identifier: MIT
"""Development only: clonotype ARI of two Singlet runs (compare_vdj's definition) with each differing cell left out in turn,
and the ARI's bootstrap spread over cells.

    python -P ari_check.py RUNS DATA LIB:TCR|BCR DIR_A DIR_B
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vdj"))
import compare_vdj as CV  # noqa: E402

runs, data, spec, da, db = sys.argv[1:6]
lib, ch = spec.split(":")
keep = CV.TCR if ch == "TCR" else CV.BCR
fca = os.path.join(data, lib, "filtered_contig_annotations.csv")
ref = CV.load_cellranger(fca)
ref = ref[ref.chain.isin(keep)]
cells = CV.cellranger_cells(fca)
gp = os.path.join(runs, lib, "gex_cells.txt")
gex = set(l.strip() for l in open(gp)) if os.path.exists(gp) else None
ca = CV._clonotypes_from_contigs(ref)
keys = {}
for d in (da, db):
    m = CV.load_singlet(os.path.join(runs, lib, d, "contigs.csv"))
    m = m[m.chain.isin(keep)]
    if gex is not None:
        m = m[m.barcode.isin(gex)]
    m = m[m.barcode.isin(cells)]
    keys[d] = CV._clonotypes_from_contigs(m)


def ari(k, drop=()):
    common = ca.index.intersection(k.index).difference(list(drop))
    return CV._pairs_same(ca[common].to_numpy(), k[common].to_numpy())["clonotype_ari"], common


ka, kb = keys[da], keys[db]
aa, ca_ = ari(ka)
ab, cb_ = ari(kb)
print(f"ARI {da} {aa:.4f}  {db} {ab:.4f}  diff {ab - aa:+.4f}  common cells {len(ca_)} {len(cb_)}")
diff = sorted(set(ka.index.symmetric_difference(kb.index)) | {b for b in ka.index.intersection(kb.index) if ka[b] != kb[b]})
print("cells that differ:", len(diff))
# Addendum 1: a drop is tolerated when it comes from cells whose new chain set equals Cell Ranger's
eqb = sum(kb.get(b) == ca.get(b) for b in diff)
eqa = sum(ka.get(b) == ca.get(b) for b in diff)
print(f"of them: new equals Cell Ranger {eqb}, old equals Cell Ranger {eqa}, neither {len(diff) - eqa - eqb}")
for b in diff:
    x, _ = ari(ka, [b])
    y, _ = ari(kb, [b])
    print(f"  without {b}: {x:.4f} {y:.4f} diff {y - x:+.4f}  ref={ca.get(b, '-')[:60]} A={ka.get(b, '-')[:60]} B={kb.get(b, '-')[:60]}")
x, _ = ari(ka, diff)
y, _ = ari(kb, diff)
print(f"without all differing cells: {x:.4f} {y:.4f}")
rng = np.random.default_rng(0)
idx = np.array(ca_)
bs = []
for _ in range(200):
    s = rng.choice(idx, len(idx))
    bs.append(CV._pairs_same(ca[s].to_numpy(), ka[s].to_numpy())["clonotype_ari"])
print(f"bootstrap over cells of {da}'s ARI: sd {np.std(bs):.4f}, 95% [{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]")

"""amb_cells.py ARTIFACT_DIR DS: where the Singlet ambient over-removal sits (per-depth-decile ratio of implied removal to truth, mass share of the worst cells)."""
import sys
import numpy as np
sys.path.insert(0, '/mnt/projects/debruinz_project/debruinz/singlet/release-gate/ambient/bench/public/capabilities')
import ambient_rna as AR
from pathlib import Path
art, ds = Path(sys.argv[1]), sys.argv[2]
df, pool = AR._load(art, ds)
d = AR._arrays(df, pool)
n, tr, imp = d['n'], d['truth'], d['singlet']['imp']
print(f'{ds}: {len(n)} pure cells, truth mean {tr.mean():.5f}, implied mean {imp.mean():.5f}, ratio {imp.mean()/tr.mean():.3f}')
print('depth decile: n range | mean truth | mean implied | ratio | share of implied mass | share of excess (imp-truth)')
q = np.quantile(n, np.linspace(0, 1, 11)); ex = (imp - tr) * n
for i in range(10):
    m = (n >= q[i]) & (n <= q[i + 1]) if i == 9 else (n >= q[i]) & (n < q[i + 1])
    print(f'  d{i}: {q[i]:7.0f}-{q[i+1]:7.0f} | {tr[m].mean():.4f} | {imp[m].mean():.4f} | {imp[m].mean()/tr[m].mean():5.2f} | {(imp[m]*n[m]).sum()/(imp*n).sum():.3f} | {ex[m].sum()/np.abs(ex).sum():+.3f}')
o = np.argsort(-(imp - tr) * n)[:15]
print('worst 15 by excess removed UMIs: n, truth, implied, excess')
for j in o: print(f'  {n[j]:7.0f} {tr[j]:.4f} {imp[j]:.4f} {(imp[j]-tr[j])*n[j]:8.1f}')
r = imp / np.maximum(tr, 1e-4)
for lo, hi in [(0, .5), (.5, 1.5), (1.5, 3), (3, 1e9)]:
    m = (r >= lo) & (r < hi); print(f'  imp/truth in [{lo},{hi}): {m.sum()} cells, mean n {n[m].mean() if m.any() else 0:.0f}, share of implied mass {(imp[m]*n[m]).sum()/(imp*n).sum():.3f}')

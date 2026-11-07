"""amb_score.py ARTIFACT_DIR DS: ambient-rna metrics of the Singlet column (and SoupX) on one barnyard library, via the gate scorer's own functions."""
import sys
import numpy as np
sys.path.insert(0, '/mnt/projects/debruinz_project/debruinz/singlet/release-gate/ambient/bench/public/capabilities')
import ambient_rna as AR
from pathlib import Path
from scipy.stats import spearmanr
art, ds = Path(sys.argv[1]), sys.argv[2]
df, pool = AR._load(art, ds)
d = AR._arrays(df, pool)
print(f'{ds}: pure cells {len(d["truth"])}  truth mean {d["truth"].mean():.5f}')
n = d['n']; low = n <= np.quantile(n, 1/3)
for t in ('singlet', 'soupx'):
    imp = d[t]['imp']; r = imp.mean() / d['truth'].mean()
    rl = imp[low].mean() / d['truth'][low].mean()
    rs = spearmanr(imp, d['truth'])[0]
    print(f'  {t:8s} mean ratio {r:.3f}  bias_abs_log2 {abs(np.log2(r)):.3f}  low-depth bias_abs_log2 {abs(np.log2(rl)):.3f}  rho_spearman {rs:.3f}  max implied {imp.max():.4f}')
imp = d['singlet']['imp']; tr = d['truth']
print(f'  singlet cells with implied > 3x truth: {int((imp > 3 * tr).sum())}')
# the gate's own cross-species residual and own-removal gap for singlet
for key in ('other_b', 'own_b', 'expected_own'):
    pass
oth_b = d['other_b']; oth_a = d['singlet'].get('other_a'); 
print('  singlet entries:', sorted(d['singlet'].keys()))

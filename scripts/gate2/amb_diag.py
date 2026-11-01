"""ambient-rna diagnosis on hgmm1k_v3 (development data only): mean implied removal vs truth per tool, by species and depth tercile."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/mnt/projects/debruinz_project/debruinz/singlet/release-gate/ambient/bench/public/capabilities')
import ambient_rna as AR
from pathlib import Path
art = Path('/mnt/projects/debruinz_project/debruinz/singlet/release-gate/ambient/artifacts/ambient-rna')
ds = 'hgmm1k_v3'
df, pool = AR._load(art, ds)
print('columns', list(df.columns))
d = AR._arrays(df, pool)
print('keys', list(d.keys()), 'n pure', len(d['truth']), 'truth mean %.5f' % d['truth'].mean())
for t in d['tools']:
    imp = d[t]['imp']
    r = imp.mean() / d['truth'].mean()
    print(f'{t:16s} mean implied {imp.mean():.5f}  ratio to truth {r:.3f}  |log2| {abs(np.log2(max(r,1e-12))):.3f}')
n = d['n']; q = np.quantile(n, [1/3, 2/3])
for name, m in (('low', n <= q[0]), ('mid', (n > q[0]) & (n <= q[1])), ('high', n > q[1])):
    print(name, 'cells', int(m.sum()), 'truth %.5f' % d['truth'][m].mean(), ' '.join(f"{t}={d[t]['imp'][m].mean()/d['truth'][m].mean():.2f}" for t in d['tools']))
H = df.human_umi.values.astype(float); M = df.mouse_umi.values.astype(float); fm = M / np.maximum(H + M, 1)
hum = fm <= AR.PURE; mou = fm >= 1 - AR.PURE
print('pure human', int(hum.sum()), 'pure mouse', int(mou.sum()), 'mouse share of pool', pool['pools']['1_100_top50k']['mouse_share'])

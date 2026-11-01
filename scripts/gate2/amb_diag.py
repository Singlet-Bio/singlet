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

# --- by species and depth decile (hgmm1k_v3, development data) ---
H = df.human_umi.values.astype(float); M = df.mouse_umi.values.astype(float); N = H + M
fm = M / np.maximum(N, 1)
pure = (fm <= AR.PURE) | (fm >= 1 - AR.PURE)
mouse_cell = (fm >= 1 - AR.PURE)[pure]
imp = d['singlet']['imp']; tr = d['truth']; n = d['n']
for name, m in (('human cells', ~mouse_cell), ('mouse cells', mouse_cell)):
    print(f'{name}: n={int(m.sum())} truth {tr[m].mean():.5f} singlet {imp[m].mean():.5f} ratio {imp[m].mean()/tr[m].mean():.3f} soupx ratio {d["soupx"]["imp"][m].mean()/tr[m].mean():.3f}')
qs = np.quantile(n, np.linspace(0, 1, 6))
for i in range(5):
    m = (n >= qs[i]) & (n <= qs[i + 1])
    print(f'depth quintile {i+1}: UMIs {qs[i]:.0f}-{qs[i+1]:.0f} n={int(m.sum())} truth {tr[m].mean():.5f} singlet {imp[m].mean():.5f} ratio {imp[m].mean()/tr[m].mean():.2f} | soupx {d["soupx"]["imp"][m].mean()/tr[m].mean():.2f}')
print('singlet implied: min %.5f p10 %.5f median %.5f p90 %.5f max %.5f; truth median %.5f p90 %.5f' % (imp.min(), *np.quantile(imp, [.1, .5, .9]), imp.max(), np.median(tr), np.quantile(tr, .9)))
print('corr(log n, log imp) singlet %.3f soupx %.3f truth %.3f' % tuple(np.corrcoef(np.log(n), np.log(np.maximum(x, 1e-6)))[0, 1] for x in (imp, d['soupx']['imp'], tr)))

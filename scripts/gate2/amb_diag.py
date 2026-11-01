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

print('--- species x depth quintile: mean singlet / mean truth, and median of per-cell ratio ---')
for sp, spm in (('human', ~mouse_cell), ('mouse', mouse_cell)):
    row = []
    for i in range(5):
        m = spm & (n >= qs[i]) & (n <= qs[i + 1])
        row.append(f'{imp[m].mean()/tr[m].mean():.2f}/{np.median(imp[m]/np.maximum(tr[m],1e-6)):.2f}(n={int(m.sum())})')
    print(sp, ' '.join(row))
print('cells with implied > 3x truth:', int((imp > 3 * tr).sum()), ' their share of total implied mass: %.2f' % (imp[imp > 3 * tr].sum() / imp.sum()))
print('mouse cells: truth mean %.5f; mean implied excluding the 20 highest implied: %.5f' % (tr[mouse_cell].mean(), np.sort(imp[mouse_cell])[:-20].mean()))
print('own_b/other_b means: other-species UMIs per pure cell %.1f  removed by singlet (implied*n) %.1f  truth*n*other_share %.1f' % (d['other_b'].mean(), (imp * n).mean(), 0))

print('--- the tail: pure cells with singlet implied > 3x truth: where does the removal come from? ---')
H_before = df.human_umi.values.astype(float)[pure]; M_before = df.mouse_umi.values.astype(float)[pure]
H_after = df.singlet_human.values.astype(float)[pure]; M_after = df.singlet_mouse.values.astype(float)[pure]
own_b = np.where(mouse_cell, M_before, H_before); oth_b = np.where(mouse_cell, H_before, M_before)
own_a = np.where(mouse_cell, M_after, H_after); oth_a = np.where(mouse_cell, H_after, M_after)
tail = imp > 3 * tr
for nm, m in (('tail', tail), ('rest', ~tail)):
    print(f'{nm}: n={int(m.sum())} mouse share {mouse_cell[m].mean():.2f} median UMIs {np.median(n[m]):.0f}; other-species UMIs before {oth_b[m].mean():.1f} after {oth_a[m].mean():.1f}; own-species UMIs before {own_b[m].mean():.1f} after {own_a[m].mean():.1f} (own removed {100*(1-own_a[m].sum()/own_b[m].sum()):.1f}% of own; other removed {100*(1-oth_a[m].sum()/oth_b[m].sum()):.1f}% of other)')
cols = [c for c in df.columns if c.startswith('singlet')]
print('singlet columns', cols)
if 'singlet_cluster' in df.columns: print(df.loc[pure].assign(tail=tail).groupby(['tail'])['singlet_cluster'].value_counts().head(20))

print('--- model rho vs implied removal (is the excess in rho or in the gene-level allocation?) ---')
rho_m = df.singlet_rho.values.astype(float)[pure]
for nm, m in (('tail', tail), ('rest', ~tail)):
    print(f'{nm}: model rho mean {rho_m[m].mean():.5f} median {np.median(rho_m[m]):.5f}; implied removal mean {imp[m].mean():.5f}; truth mean {tr[m].mean():.5f}')

print('--- clusters of the tail cells ---')
ac = pd.read_csv('/mnt/projects/debruinz_project/debruinz/singlet/release-gate/gate2/pb/runs/hgmm1k_v3/singlet-public/ambient_contamination.tsv', sep=chr(9))
print('ambient_contamination.tsv columns', list(ac.columns))
bc_col = ac.columns[0]
ac = ac.set_index(bc_col)
pb = df.barcode.values[pure]
ccol = next((c for c in ac.columns if 'cluster' in c.lower()), None)
if ccol is not None:
    cl = ac.reindex(pb)[ccol].values
    tmp = pd.DataFrame({'cluster': cl, 'mouse': mouse_cell, 'tail': tail, 'n': n, 'rho': ac.reindex(pb)[ac.columns[0]].values if False else rho_m})
    print(tmp.groupby('cluster').agg(cells=('mouse', 'size'), mouse_frac=('mouse', 'mean'), tail=('tail', 'sum'), median_umi=('n', 'median'), rho_med=('rho', 'median')))
    print('tail cells by cluster:', tmp[tmp.tail].cluster.value_counts().to_dict())

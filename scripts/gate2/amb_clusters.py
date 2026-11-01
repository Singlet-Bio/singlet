"""amb_clusters.py RUN_DIR ARTIFACT_DIR DS: per Singlet ambient cluster, cells / depth / mean truth / mean implied removal / share of excess."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/mnt/projects/debruinz_project/debruinz/singlet/release-gate/ambient/bench/public/capabilities')
import ambient_rna as AR
from pathlib import Path
run, art, ds = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
df, pool = AR._load(art, ds)
d = AR._arrays(df, pool)
a = pd.read_csv(run / 'ambient_contamination.tsv', sep='\t')
a['barcode'] = a.barcode.astype(str).str.replace(r'-1$', '', regex=True)
cl = a.set_index('barcode').cluster
H, M = df.human_umi.values.astype(float), df.mouse_umi.values.astype(float)
fm = M / np.maximum(H + M, 1)
pure = (fm <= AR.PURE) | (fm >= 1 - AR.PURE)
for t in AR.TOOLS:
    if f"{t}_human" in df and not df.loc[pure, f"{t}_human"].notna().all():
        pure &= df[f"{t}_human"].notna().values
assert pure.sum() == len(d['truth']), (pure.sum(), len(d['truth']))
b = df.barcode.astype(str).str.replace(r'-1$', '', regex=True).values[pure]
k = cl.reindex(b).values
t, imp, n = d['truth'], d['singlet']['imp'], d['n']
mouse = fm[pure] >= 0.5
print('cluster | species(mouse frac) | cells | mean depth | mean truth | mean implied | ratio | share of excess removed UMIs | cells imp>3x truth')
ex = (imp - t) * n
for c in sorted(set(k[~pd.isna(k)])):
    m = k == c
    print(f'  {int(c)} | {mouse[m].mean():.2f} | {m.sum():4d} | {n[m].mean():7.0f} | {t[m].mean():.4f} | {imp[m].mean():.4f} | {imp[m].mean()/t[m].mean():5.2f} | {ex[m].sum()/np.abs(ex).sum():+.3f} | {(imp[m] > 3*t[m]).sum()}')
print('median rho per cluster from the run:'); print(a.groupby('cluster').rho.agg(['count', 'median', 'max']).round(4).to_string())

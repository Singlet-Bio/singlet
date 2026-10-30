"""Singlet non-host calls vs Kraken2 (and a minimap2/SARS-CoV-2 oracle) on mock / infected / PBMC samples.
usage: python nonhost_compare.py BENCH_DIR OUT_JSON
BENCH_DIR layout (see the .sbatch files):
  singlet/<label>/nonhost/{nonhost_summary.json,nonhost_em_abundance.tsv,nonhost_per_cell.tsv}   (mock, dpi2)
  <PBMC>: the existing t4_nonhost_r1 run is passed through env PBMC_SINGLET
  kraken/<label>/{percell_conf0.0.tsv,percell_conf0.1.tsv,species_conf0.0.tsv}
  oracle/<label>/percell.tsv  (barcode, reads, umis)  -- optional
  cells/<label>.txt  -- called-cell barcodes (STAR Solo filtered)
"""
import sys, os, json, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

B, OUT = sys.argv[1:3]
SARS = 'Severe acute respiratory syndrome'
R = {}


def rd(p, **k):
    return pd.read_csv(p, sep='\t', **k) if os.path.exists(p) else None


def cells(l):
    return [x.strip().split('\t')[0] for x in open(f'{B}/cells/{l}.txt') if x.strip()]


def singlet_dir(l):
    return os.environ.get('PBMC_SINGLET') if l == 'pbmc1k_v3' else f'{B}/singlet/{l}'


def kraken_umis(l, conf, sars_only=False):
    d = rd(f'{B}/kraken/{l}/percell_conf{conf}.tsv')
    if d is None:
        return None
    if sars_only:
        d = d[d['name'].str.contains(SARS)]
    return d.groupby('barcode')['umis'].sum()


def singlet_umis(l, sars_only=False):
    d = rd(f'{singlet_dir(l)}/nonhost/nonhost_per_cell.tsv')
    if d is None:
        return None
    d = d[d['kingdom'] == 'VIRAL']
    if sars_only:
        d = d[d['species_name'].astype(str).str.contains(SARS) | (d['species_taxon_id'] == 8877)]  # 8877 = NC_045512.2 in Singlet's viral index
    return d.groupby('barcode')['n_umis'].sum()


def oracle_umis(l):
    d = rd(f'{B}/oracle/{l}/percell.tsv')
    return None if d is None else d.groupby('barcode')['umis'].sum()


# ---- sample level ----
R['sample'] = {}
for l in ('mock', 'dpi2', 'pbmc1k_v3'):
    r = {'n_cells': len(cells(l)) if os.path.exists(f'{B}/cells/{l}.txt') else None}
    sj = f'{singlet_dir(l)}/nonhost/nonhost_summary.json'
    if os.path.exists(sj):
        r['singlet_summary'] = json.load(open(sj))
    em = rd(f'{singlet_dir(l)}/nonhost/nonhost_em_abundance.tsv')
    r['singlet_em_species'] = [] if em is None else em.head(8).to_dict('records')
    pc = rd(f'{singlet_dir(l)}/nonhost/nonhost_per_cell.tsv')
    r['singlet_per_cell_file'] = None if pc is None else {
        'rows': len(pc), 'species': pc.groupby(['kingdom', 'species_name'])['n_umis'].sum().sort_values(ascending=False).head(8).to_dict().__repr__()}
    for conf in ('0.0', '0.1'):
        sp = rd(f'{B}/kraken/{l}/species_conf{conf}.tsv')
        r[f'kraken_conf{conf}_top'] = None if sp is None else sp.head(6).to_dict('records')
    R['sample'][l] = r

# ---- per-cell separation: mock (0) vs dpi2 (1), cells called in each sample ----
def auroc(fn):
    vals, y = [], []
    for l, lab in (('mock', 0), ('dpi2', 1)):
        s = fn(l)
        if s is None:
            return None
        for c in cells(l):
            vals.append(float(s.get(c, 0))); y.append(lab)
    vals, y = np.array(vals), np.array(y)
    return {'n_cells': int(len(y)), 'AUROC': float(roc_auc_score(y, vals)),
            'frac_cells_ge1_UMI_mock': float((vals[y == 0] > 0).mean()), 'frac_cells_ge1_UMI_dpi2': float((vals[y == 1] > 0).mean()),
            'frac_cells_ge3_UMI_mock': float((vals[y == 0] >= 3).mean()), 'frac_cells_ge3_UMI_dpi2': float((vals[y == 1] >= 3).mean()),
            'total_UMI_mock': float(vals[y == 0].sum()), 'total_UMI_dpi2': float(vals[y == 1].sum())}


R['separation'] = {
    'singlet_viral_all': auroc(lambda l: singlet_umis(l)),
    'singlet_SARSr': auroc(lambda l: singlet_umis(l, True)),
    'kraken0.0_viral_all': auroc(lambda l: kraken_umis(l, '0.0')),
    'kraken0.0_SARSr': auroc(lambda l: kraken_umis(l, '0.0', True)),
    'kraken0.1_SARSr': auroc(lambda l: kraken_umis(l, '0.1', True)),
    'oracle_minimap2_NC_045512.2': auroc(oracle_umis)}

# ---- per-cell agreement in dpi2 ----
ag = {}
cs = cells('dpi2') if os.path.exists(f'{B}/cells/dpi2.txt') else []
vecs = {'singlet': singlet_umis('dpi2', True), 'kraken0.0': kraken_umis('dpi2', '0.0', True),
        'kraken0.1': kraken_umis('dpi2', '0.1', True), 'oracle': oracle_umis('dpi2')}
for a in vecs:
    for b in vecs:
        if a < b and vecs[a] is not None and vecs[b] is not None:
            x = np.array([vecs[a].get(c, 0) for c in cs], float); y = np.array([vecs[b].get(c, 0) for c in cs], float)
            ag[f'{a}_vs_{b}'] = {'spearman': float(spearmanr(x, y)[0]) if x.std() and y.std() else None,
                                 'cells_pos_a': int((x > 0).sum()), 'cells_pos_b': int((y > 0).sum()), 'cells_pos_both': int(((x > 0) & (y > 0)).sum())}
R['dpi2_per_cell_agreement'] = ag
# agreement with the minimap2 oracle on "cell has >= k SARS-CoV-2 UMIs" (same k for tool and oracle)
if vecs['oracle'] is not None:
    pr = {}
    for k in (1, 2, 3):
        o = np.array([vecs['oracle'].get(c, 0) for c in cs], float) >= k
        for name in ('singlet', 'kraken0.0', 'kraken0.1'):
            if vecs[name] is None:
                continue
            p = np.array([vecs[name].get(c, 0) for c in cs], float) >= k
            pr[f'{name}_k>={k}'] = {'oracle_pos': int(o.sum()), 'called_pos': int(p.sum()), 'TP': int((p & o).sum()),
                                    'precision': float((p & o).sum() / max(1, p.sum())), 'recall': float((p & o).sum() / max(1, o.sum()))}
    R['dpi2_vs_oracle'] = pr
# ---- PBMC: false positives ----
pb = {}
for name, v in (('singlet_viral', singlet_umis('pbmc1k_v3')), ('singlet_SARSr', singlet_umis('pbmc1k_v3', True)), ('kraken0.0_viral', kraken_umis('pbmc1k_v3', '0.0')), ('kraken0.1_viral', kraken_umis('pbmc1k_v3', '0.1'))):
    if v is not None and os.path.exists(f'{B}/cells/pbmc1k_v3.txt'):
        c = cells('pbmc1k_v3')
        pb[name] = {'cells': len(c), 'cells_with_ge1_UMI': int(sum(v.get(x, 0) > 0 for x in c)), 'total_UMI_in_cells': float(sum(v.get(x, 0) for x in c))}
R['pbmc_false_positives'] = pb
json.dump(R, open(OUT, 'w'), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))

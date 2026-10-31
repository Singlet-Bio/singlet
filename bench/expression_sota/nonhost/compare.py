"""Singlet-side viral gene counts (prototype) vs STARsolo reference vs Kraken2 / minimap2 oracle.  usage: python -I compare.py OUT.json
All comparisons are over Singlet's called cells (cells/gse166766_hbec_dpi2.txt); reference UMIs outside those cells are reported but cannot be recovered."""
import sys, os, json, gzip, collections
import numpy as np, pandas as pd
from scipy.io import mmread
from scipy.stats import spearmanr, pearsonr

W = '/mnt/projects/debruinz_project/debruinz/singlet'
X = f'{W}/expression-sota/nonhost'
GATE = f'{W}/release-gate/perf/nonhost'
cells = [l.strip() for l in open(f'{GATE}/cells/gse166766_hbec_dpi2.txt') if l.strip()]
C = set(cells)
trs = json.load(open(f'{X}/ref/viral/viral.trs.json'))
VG = list(trs['genes'])
out = {}


def viral_counts(d, feat='Gene'):
    """{(barcode, gene): umis} for viral genes from a Solo.out raw matrix."""
    r = f'{d}/Solo.out/{feat}/raw'
    bc = [l.strip() for l in open(f'{r}/barcodes.tsv')]
    ft = [l.rstrip('\n').split('\t') for l in open(f'{r}/features.tsv')]
    idx = {i: f[0] for i, f in enumerate(ft) if f[0] in trs['genes']}
    m = mmread(f'{r}/matrix.mtx').tocoo()
    res = {}
    for i, j, v in zip(m.row, m.col, m.data):
        if i in idx:
            res[(bc[j], idx[i])] = int(v)
    return res


def table(d, subset=True):
    return {k: v for k, v in d.items() if (k[0] in C or not subset)}


def vec(d, genes=None):
    """per-cell total over given genes, ordered as `cells`"""
    t = collections.Counter()
    for (b, g), v in d.items():
        if b in C and (genes is None or g in genes):
            t[b] += v
    return np.array([t[c] for c in cells], float)


def corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return {'spearman': None, 'pearson': None}
    return {'spearman': round(float(spearmanr(a, b)[0]), 4), 'pearson': round(float(pearsonr(a, b)[0]), 4)}


ref = {f: viral_counts(f'{X}/ref_starsolo/dpi2', f) for f in ('Gene', 'GeneFull')}
out['ref_umis_all_barcodes_Gene'] = sum(ref['Gene'].values())
out['ref_umis_all_barcodes_GeneFull'] = sum(ref['GeneFull'].values())
R = table(ref['Gene'])
out['ref_umis_in_called_cells_Gene'] = sum(R.values())
out['ref_umis_in_called_cells_GeneFull'] = sum(table(ref['GeneFull']).values())
out['ref_cells_ge1'] = len({b for b, g in R})
out['ref_cells_ge3'] = int(sum(vec(R) >= 3))
out['ref_umis_by_gene'] = {g: int(sum(v for (b, gg), v in R.items() if gg == g)) for g in VG}
try:
    fb = {l.strip() for l in open(f'{X}/ref_starsolo/dpi2/Solo.out/Gene/filtered/barcodes.tsv')}
    out['starsolo_EmptyDrops_cells'] = len(fb); out['called_cells'] = len(C); out['called_cells_in_EmptyDrops'] = len(fb & C)
except Exception as e:
    out['filtered_err'] = str(e)


def compare(name, P, extra=None):
    P = table(P)
    r = {}
    ur, up = sum(R.values()), sum(P.values())
    keys = set(R) | set(P)
    ov = sum(min(R.get(k, 0), P.get(k, 0)) for k in keys)
    r['proto_umis'] = up
    r['umi_sensitivity_overlap'] = round(ov / ur, 4)          # sum min(ref,proto) / sum ref over (cell, gene)
    r['umi_ratio_total'] = round(up / ur, 4)
    rc, pc = {b for b, g in R}, {b for b, g in P}
    r['cell_sensitivity_ge1'] = round(len(rc & pc) / len(rc), 4)
    r['cells_proto_ge1'] = len(pc); r['cells_ref_ge1'] = len(rc)
    rp, pp = set(R), set(P)
    r['cellgene_pairs_ref'] = len(rp); r['cellgene_pairs_proto'] = len(pp)
    r['cellgene_recall'] = round(len(rp & pp) / len(rp), 4)
    r['cellgene_precision_vs_ref'] = round(len(rp & pp) / len(pp), 4) if pp else None   # proto calls absent from reference = 1 - this
    r['cell_precision_vs_ref'] = round(len(rc & pc) / len(pc), 4) if pc else None
    r['umi_excess_over_ref_frac'] = round(sum(max(P.get(k, 0) - R.get(k, 0), 0) for k in keys) / up, 4) if up else None
    r['total_viral_per_cell'] = corr(vec(R), vec(P))
    r['per_gene'] = {}
    for g in VG:
        a, b = vec(R, {g}), vec(P, {g})
        r['per_gene'][g] = {'ref_umis': int(a.sum()), 'proto_umis': int(b.sum()), 'ref_cells': int((a > 0).sum()), 'proto_cells': int((b > 0).sum()),
                            **corr(a, b)}
    # pooled over the cell x gene table
    a = np.array([R.get(k, 0) for k in sorted(keys)], float); b = np.array([P.get(k, 0) for k in sorted(keys)], float)
    r['cellgene_table_corr_nonzero_union'] = corr(a, b)
    if extra: r.update(extra)
    out[name] = r


for lab in ('kept', 'all_unmapped'):
    d = f'{X}/proto/dpi2/{lab}'
    if os.path.exists(f'{d}/Solo.out/Gene/raw/matrix.mtx'):
        compare('proto_' + lab, viral_counts(d, 'Gene'))

# Kraken2 (viral db, gate run on the same STAR-unmapped reads) and minimap2 oracle: per-cell, species level only
kd = f'{GATE}/gse166766_hbec_dpi2/kraken2_viral/percell.tsv.gz'
k = pd.read_csv(kd, sep='\t')
ks = k[k.taxid == 694009].groupby('barcode').umis.sum()
kv = k.groupby('barcode').umis.sum()
o = pd.read_csv(f'{GATE}/gse166766_hbec_dpi2/oracle/percell.tsv', sep='\t').set_index('barcode').umis
rv = vec(R)
for name, s in (('kraken2_SARSr', ks), ('kraken2_all_viral', kv), ('minimap2_oracle', o)):
    v = np.array([s.get(c, 0) for c in cells], float)
    pc = {c for c in cells if s.get(c, 0) > 0}; rc = {b for b, g in R}
    out[name] = {'umis_in_called_cells': int(v.sum()), 'cells_ge1': len(pc), 'cell_sensitivity_vs_ref': round(len(rc & pc) / len(rc), 4),
                 'cell_precision_vs_ref': round(len(rc & pc) / len(pc), 4), 'umi_ratio_to_ref': round(v.sum() / rv.sum(), 4),
                 'total_viral_per_cell_vs_ref': corr(rv, v), 'per_gene_resolution': 'none (taxon only)'}
# vs the Singlet screen's own per-cell species table (engine output)
sp = pd.read_csv(f'{GATE}/gse166766_hbec_dpi2/singlet/nonhost/nonhost_per_cell.tsv', sep='\t')
sp = sp[sp.species_taxon_id == 8877].groupby('barcode').n_umis.sum()
v = np.array([sp.get(c, 0) for c in cells], float)
out['singlet_species_table_SARS2'] = {'umis': int(v.sum()), 'cells_ge1': int((v > 0).sum()), 'total_viral_per_cell_vs_ref': corr(rv, v), 'per_gene_resolution': 'none'}

# alignment-level reference (any viral-genome alignment with a corrected CB/UB, whether or not it overlaps a gene): the fair target for taxon-level tools
def aln_umis(sam):
    s = collections.defaultdict(set)
    for l in open(sam):
        if l[0] == '@':
            continue
        f = l.rstrip('\n').split('\t')
        if int(f[1]) & 0x104:
            continue
        t = {x[:2]: x[5:] for x in f[11:]}
        if t.get('CB', '-') in C and t.get('UB', '-') != '-':
            s[t['CB']].add(t['UB'])
    return {c: len(u) for c, u in s.items()}


hist = collections.Counter()   # where viral alignments start (1 kb bins) in the reference run
for l in open(f'{X}/ref_starsolo/dpi2/viral_aligned.sam'):
    if l[0] != '@':
        f = l.split('\t', 4)
        if not int(f[1]) & 0x104:
            hist[int(f[3]) // 1000 * 1000] += 1
out['ref_viral_alignment_start_hist_1kb'] = dict(sorted(hist.items()))
a3 = sum(v for k, v in hist.items() if k >= 29000)
out['ref_alignments_start_ge_29000_frac'] = round(a3 / sum(hist.values()), 4)
ref_aln = aln_umis(f'{X}/ref_starsolo/dpi2/viral_aligned.sam')
ra = np.array([ref_aln.get(c, 0) for c in cells], float)
out['ref_alignment_level'] = {'umis_in_called_cells': int(ra.sum()), 'cells_ge1': int((ra > 0).sum()), 'gene_assigned_umi_fraction': round(rv.sum() / ra.sum(), 4)}
for name, s in (('proto_kept', aln_umis(f'{X}/proto/dpi2/kept/aligned.sam')), ('proto_all_unmapped', aln_umis(f'{X}/proto/dpi2/all_unmapped/aligned.sam')),
                ('kraken2_SARSr', ks), ('kraken2_all_viral', kv), ('minimap2_oracle', o), ('singlet_species_table_SARS2', sp)):
    v = np.array([s.get(c, 0) for c in cells], float)
    pc = {c for c in cells if s.get(c, 0) > 0}; rc = {c for c in cells if ref_aln.get(c, 0) > 0}
    out['vs_alignment_level_' + name] = {'umis': int(v.sum()), 'cells_ge1': len(pc), 'cell_recall': round(len(rc & pc) / len(rc), 4),
                                         'cell_precision': round(len(rc & pc) / len(pc), 4), 'umi_ratio': round(v.sum() / ra.sum(), 4), **corr(ra, v)}

# read level: Kraken2 species call vs STAR viral alignment on the same STAR-unmapped reads
sam = f'{X}/proto/dpi2/all_unmapped/aligned.sam'
if os.path.exists(sam):
    al = {l.split('\t')[0] for l in open(sam) if l[0] != '@'}
    kr = set()
    kr_all = 0
    with gzip.open(f'{GATE}/gse166766_hbec_dpi2/kraken2_viral/out.txt.gz', 'rt') as f:
        for l in f:
            p = l.split('\t')
            if p[0] == 'C':
                kr_all += 1
                if p[2] in ('694009', '2697049', '2509486'):  # SARSr-CoV, SARS-CoV-2 (and its parent clades in some DBs)
                    kr.add(p[1])
    out['read_level'] = {'star_viral_aligned_reads': len(al), 'kraken_sars_reads': len(kr), 'kraken_classified_any_viral': kr_all,
                         'overlap': len(al & kr), 'kraken_recall_of_star_aligned': round(len(al & kr) / max(1, len(al)), 4),
                         'star_recall_of_kraken_sars': round(len(al & kr) / max(1, len(kr)), 4)}
    kept = {l[1:].split()[0] for i, l in enumerate(open(f'{X}/proto/dpi2/kept.cdna.fq')) if i % 4 == 0}
    out['read_level']['star_aligned_in_screen_kept_reads'] = len(al & kept)
    out['read_level']['star_aligned_dropped_by_screen'] = len(al - kept)

# sgRNA (leader-TRS junction) concordance
def sg(path):
    d = pd.read_csv(path, sep='\t')
    d = d[d.barcode.isin(C)]
    return d


try:
    rs, ps = sg(f'{X}/ref_starsolo/dpi2/sgrna.percell.tsv'), sg(f'{X}/proto/dpi2/kept/sgrna.percell.tsv')
    res = {}
    for ev in ('all', 'spliced', 'clipped'):
        a = rs if ev == 'all' else rs[rs.evidence == ev]
        b = ps if ev == 'all' else ps[ps.evidence == ev]
        ra = a.groupby(['barcode', 'sgrna']).umis.sum(); pb = b.groupby(['barcode', 'sgrna']).umis.sum()
        rk, pk = set(ra.index), set(pb.index)
        res[ev] = {'ref_umis': int(ra.sum()), 'proto_umis': int(pb.sum()), 'ref_cells': len({x[0] for x in rk}), 'proto_cells': len({x[0] for x in pk}),
                   'cell_sgrna_pairs_ref': len(rk), 'cell_sgrna_pairs_proto': len(pk), 'pair_recall': round(len(rk & pk) / max(1, len(rk)), 4),
                   'pair_precision': round(len(rk & pk) / max(1, len(pk)), 4),
                   'umi_sensitivity_overlap': round(sum(min(ra.get(x, 0), pb.get(x, 0)) for x in rk | pk) / max(1, ra.sum()), 4),
                   'ref_by_sgrna': ra.groupby(level=1).sum().to_dict(), 'proto_by_sgrna': pb.groupby(level=1).sum().to_dict()}
    out['sgrna'] = res
except Exception as e:
    out['sgrna_err'] = repr(e)
json.dump(out, open(sys.argv[1], 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))

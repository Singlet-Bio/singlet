"""Idea 3: turn results/idea3_a_measure.jsonl, results/idea3_b_measure.jsonl and the codec TSVs into the markdown
tables of notes/idea3.md.  Usage: python -P tools/idea3_report.py > results/idea3_tables.md"""
import sys, os, json, csv
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R = lambda *p: os.path.join(ROOT, *p)

def jsonl(path):
    out = {}
    if not os.path.exists(path):
        return out
    for l in open(path):
        l = l.strip()
        if l:
            r = json.loads(l)
            out[(r['sample'], r.get('kind', ''))] = r        # last row wins
    return out

ORDER = ['pbmc_v2_typical', 'pbmc_v2_shallow', 'pbmc_v3_lowgenes', 'tcell_5p', 'tumor_3p', 'brain_sn_v3_deep', 'brain_sn_v3_intronic']

def ref_payload():
    """ref_1pz_v1_L3 matrix chunk payload (bits/nnz) from Track A's accounting."""
    d = {}
    p = R('results', 'track_a_account_files.tsv')
    if os.path.exists(p):
        for r in csv.DictReader(open(p), delimiter='\t'):
            d[(r['sample'], r['member'].split('_')[0])] = int(r['chunk_payload']) * 8.0 / int(r['nnz'])
    return d

def table(hdr, rows, fmt=None):
    out = ['| ' + ' | '.join(hdr) + ' |', '|' + '|'.join('---' for _ in hdr) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)

def f(x, d=2):
    return '-' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))

def section_a(A, ref):
    out = []
    # ---- structure
    rows = []
    for s in ORDER:
        for k in ('exon', 'intron'):
            r = A.get((s, k))
            if not r: continue
            rows.append([s, k, r['m'], r['G'], f(r['ivl_per_gene_mean'], 1), f(r['ivl_per_gene_med'], 0), f(r['ivl_per_gene_p90'], 0), r['ivl_per_gene_max'],
                         f(r['umi_weighted_ivl_per_gene'], 1), f(100 * r['frac_umi_single_ivl_genes'], 1), r['nnz_int'], r['nnz_gene'], f(r['nnz_int_per_nnz_gene'], 3), f(r['umi_per_nnz_int'], 2), f(100 * r['frac_umi_c1'], 1)])
    out.append('### A1. Gene / interval structure\n')
    out.append('intervals per gene: mean / median / p90 / max over the reference; "UMI-wtd" = mean intervals of the gene a random UMI falls in; "1-ivl %" = share of UMIs in single-interval genes (no split to code); "c=1 %" = share of UMIs in (cell, gene) pairs with exactly one UMI.\n')
    out.append(table(['sample', 'kind', 'rows m', 'genes G', 'ivl/gene', 'med', 'p90', 'max', 'UMI-wtd', '1-ivl %', 'nnz_int', 'nnz_gene', 'nnz_int/nnz_gene', 'UMI/nnz_int', 'c=1 %'], rows))
    # ---- H(interval | gene)
    rows = []
    for s in ORDER:
        for k in ('exon', 'intron'):
            r = A.get((s, k))
            if not r: continue
            U = r['U']; nz = r['nnz_int']
            best_flat = min(v for kk, v in r.items() if kk.startswith('split_adapt_flat_'))
            best_prior = min([v for kk, v in r.items() if kk.startswith('split_adapt_oosbestprior_')] or [None]) if 'split_oosbest_iid_bits' in r else None
            rows.append([s, k, f(r['H_I_given_G_in_bits_per_umi'], 3), f(r['H_I_given_G_cbucket_in_bits_per_umi'], 3),
                         f(r['H_I_given_G_in_c1'], 2), f(r['H_I_given_G_in_c5p'], 2),
                         f(r['split_in_multiset_bits'] / U, 3),
                         f(r.get('split_oos_a0.25_iid_bits', 0) / U, 3) if 'split_oos_a0.25_iid_bits' in r else '-',
                         f(r['split_oosbest_iid_bits'] / U, 3) if 'split_oosbest_iid_bits' in r else '-', r.get('split_oosbest_id', '-'),
                         f(r['split_len_iid_bits'] / U, 3),
                         f(best_flat / U, 3), f(best_prior / U, 3) if best_prior else '-',
                         f(best_flat / nz, 3)])
    out.append('\n### A2. Cost of the interval choice, bits per UMI (H(interval | gene))\n')
    out.append('in = in-sample static per-gene interval table (optimistic: table not paid for); in|c = also conditioned on the (cell,gene) UMI count bucket {1, 2, 3-4, 5+}; c=1 / c>=5 = the per-bucket values; '
               'multiset = in-sample static but coding the split as an exact multiset (what a codec has to transmit; smaller than iid because the order of UMIs within a gene is not information); '
               'pool = static table pooled over all other studies of the same build; best1 = the single best other sample (oracle choice); length = zero-shot prior proportional to interval length; '
               'adapt = adaptive Dirichlet-multinomial from a flat prior (self-contained: no table transmitted, no model); adapt+best1 = adaptive from the best1 prior. Last column: adapt in bits per interval nonzero.\n')
    out.append(table(['sample', 'kind', 'in', 'in|c', 'c=1', 'c>=5', 'in multiset', 'pool', 'best1', 'best1 id', 'length', 'adapt', 'adapt+best1', 'adapt b/nnz'], rows))
    # ---- level comparison
    rows = []
    for s in ORDER:
        for k in ('exon', 'intron'):
            r = A.get((s, k))
            if not r: continue
            nz = r['nnz_int']
            best_flat = min(v for kk, v in r.items() if kk.startswith('split_adapt_flat_'))
            int_in = r['int_kbits'] + r['int_pres_in_cls'] + r['int_cnt_in_cls'] + r['int_cls_side']
            gene_in = r['gene_kbits'] + r['gene_pres_in_cls'] + r['gene_cnt_in_cls'] + r['gene_cls_side'] + best_flat
            rows.append([s, k, f(ref.get((s, k)), 2),
                         f(int_in / nz, 2), f(gene_in / nz, 2),
                         f(r['gene_pres_in_cls'] / nz, 2), f(r['gene_cnt_in_cls'] / nz, 2), f(best_flat / nz, 2), f(r['gene_cls_side'] / nz, 2),
                         f((r['int_pres_oos_m30'] + r['int_cnt_oos_cls'] + r['int_kbits']) / nz, 2) if 'int_pres_oos_m30' in r else '-',
                         f((r['gene_pres_oos_m30'] + r['gene_cnt_oos_cls'] + r['gene_kbits'] + best_flat) / nz, 2) if 'gene_pres_oos_m30' in r else '-',
                         f((r['int_pres_oos_cls_best'] + r['int_cnt_oos_cls_best'] + r['int_kbits']) / nz, 2) if 'int_pres_oos_cls_best' in r else '-',
                         f((r['gene_pres_oos_cls_best'] + r['gene_cnt_oos_cls_best'] + r['gene_kbits'] + best_flat) / nz, 2) if 'gene_pres_oos_cls_best' in r else '-',
                         r.get('gene_pres_oos_cls_best_id', '-')])
    out.append('\n### A3. Interval-level vs gene-level coding, bits per interval nonzero (cross-entropies, no coder overhead)\n')
    out.append('ref = today\'s .1pz matrix chunk payload (zstd-3 VOCSC, Track A accounting). "int" = per-row Bernoulli presence x depth-bucket + per-row count model, i.e. the direct interval-level model; '
               '"gene" = the same model at gene level + the adaptive interval split. in-sample(cls) = per-row rates quantised to 64 rate classes, class x bucket tables transmitted (side column counts them, '
               'and the per-gene class list is in "side"); oos(pool) = per-feature x bucket tables pooled over the other studies, shrunk (m=30) to the class rate; oos(best1) = class-only tables from the single '
               'best other sample (class list from that sample, nothing transmitted). The gene-level columns P / C / S / side split the in-sample total into presence / counts / split / side info.\n')
    out.append(table(['sample', 'kind', 'ref', 'int in', 'gene in', 'P', 'C', 'S', 'side', 'int oos pool', 'gene oos pool', 'int oos best1', 'gene oos best1', 'best1 id'], rows))
    # ---- question C: which gene vs which interval
    rows = []
    for s in ORDER:
        for k in ('exon', 'intron'):
            r = A.get((s, k))
            if not r: continue
            nz = r['nnz_int']
            best_flat = min(v for kk, v in r.items() if kk.startswith('split_adapt_flat_'))
            tot = r['gene_kbits'] + r['gene_pres_in_cls'] + r['gene_cnt_in_cls'] + best_flat
            rows.append([s, k, f(r['gene_pres_in_cls'] / nz, 2), f(best_flat / nz, 2), f(r['gene_cnt_in_cls'] / nz, 2), f(100 * best_flat / tot, 1), f(100 * r['split_in_multiset_bits'] / tot, 1),
                         f((r['int_pres_in_cls'] - r['gene_pres_in_cls']) / nz, 2)])
    out.append('\n### A4. Where the bits go in the factorised model (in-sample, bits per interval nonzero)\n')
    out.append('which gene = gene presence; which interval = adaptive split; how many = gene count given presence. "split %" = share of the total that the interval choice costs; "static split %" = the same with an in-sample static table (lower bound). '
               'last column = extra presence bits the flat interval-level model pays over the gene-level presence (the price of treating every interval as an independent feature).\n')
    out.append(table(['sample', 'kind', 'which gene', 'which interval', 'how many', 'split %', 'static split %', 'int-gene presence'], rows))
    return '\n'.join(out)

def section_b(Bm):
    out = ['### B1. Exon -> intron information at gene level (per cell), 2-fold cross-validated (cells by parity)\n']
    out.append('All costs in bits per intron gene-nonzero. A = P(intron present | gene rate class, intron depth bucket); +e = also conditioned on the exon count bucket of the same gene in the same cell {0, 1, 2-3, 4+}; '
               '+cE = also on the exon depth class of the cell; MI = raw mutual information I(exon bucket; intron presence) without conditioning (bits per intron gene-nnz); P(I|e) = P(intron present | e bucket); '
               'cnt = count given presence; k = bits for the per-cell intron gene count, alone vs residual against round(ratio x k_exon); "w/ exon %" = share of intron gene-nonzeros whose gene has exon UMIs in the same cell.\n')
    rows = []
    for s in ORDER:
        r = Bm.get((s, ''))
        if not r: continue
        nz = r['nnz_gene_I']
        rows.append([s, nz, f(r['pres_bits_A_cls_b'] / nz, 3), f(r['pres_bits_B_cls_b_e'] / nz, 3), f(r['pres_bits_C_cls_b_cE'] / nz, 3), f(r['pres_bits_D_cls_b_cE_e'] / nz, 3),
                     f((r['pres_bits_A_cls_b'] - r['pres_bits_B_cls_b_e']) / nz, 3), f((r['pres_bits_C_cls_b_cE'] - r['pres_bits_D_cls_b_cE_e']) / nz, 3),
                     f(r['MI_raw_bits_total'] / nz, 3), ' / '.join(f'{p:.3f}' for p in r['P_I_given_e']),
                     f(r['cnt_bits_A'] / nz, 3), f(r['cnt_bits_B_e'] / nz, 3), f(r['k_bits_I_alone'] / r['n'], 2), f(r['k_bits_I_given_E_ratio'] / r['n'], 2), f(r['corr_log_kE_kI'], 2),
                     f(100 * r['frac_intron_nz_with_exon'], 1)])
    out.append(table(['sample', 'nnz_gene_I', 'A', 'A+e', 'A+cE', 'A+cE+e', 'gain e|A', 'gain e|A+cE', 'MI raw', 'P(I|e=0/1/2-3/4+)', 'cnt', 'cnt+e', 'k alone', 'k|kE', 'corr', 'w/ exon %'], rows))
    return '\n'.join(out)

def tsv_rows(path):
    if not os.path.exists(path): return []
    return list(csv.DictReader(open(path), delimiter='\t'))

def section_codec(ref):
    out = []
    rows = tsv_rows(R('results', 'idea3_run.tsv'))
    if rows:
        out.append('### A5. Codec results (idea3_bench, payload = range-coded matrix stream; names zstd-9 separately)\n')
        out.append('bits per interval nonzero. K/P/C/S/side = per-cell nnz, gene presence, count, interval split, transmitted tables. ref = today\'s chunk payload. dec = single-thread decode, min of reps, on a machine shared with ~7 other agents.\n')
        t = []
        for r in rows:
            nz = float(r['nnz'])
            t.append([r['sample'], r['member'].split('_')[0], r['codec'], f(ref.get((r['sample'], r['member'].split('_')[0])), 2), f(float(r['payload_bits_per_nnz']), 3),
                      f(float(r['k_bits']) / nz, 3), f(float(r['pres_bits']) / nz, 3), f(float(r['cnt_bits']) / nz, 3), f(float(r['split_bits']) / nz, 3), f(float(r['side_bytes']) * 8 / nz, 3),
                      f(float(r['model_bytes']) / 1e6, 2), f(float(r['enc_ms']) / 1e3, 1), f(float(r['dec_ms_min']) / 1e3, 2), f(float(r['dec_Mnnz_s']), 2), r['matrix_ok']])
        out.append(table(['sample', 'kind', 'codec', 'ref', 'payload b/nnz', 'K', 'P', 'C', 'S', 'side', 'model MB', 'enc s', 'dec s', 'Mnnz/s', 'ok'], t))
    rows = tsv_rows(R('results', 'idea3_joint.tsv'))
    if rows:
        out.append('\n### B2. Joint exon + intron codec results (idea3_joint)\n')
        t = []
        for r in rows:
            nz = float(r['nnz'])
            t.append([r['sample'], r['member'].split('_')[0], r['codec'], f(ref.get((r['sample'], r['member'].split('_')[0])), 2), f(float(r['payload_bits_per_nnz']), 3),
                      f(float(r['k_bits']) / nz, 3), f(float(r['pres_bits']) / nz, 3), f(float(r['cnt_bits']) / nz, 3), f(float(r['split_bits']) / nz, 3), f(float(r['side_bytes']) * 8 / nz, 3),
                      f(float(r['dec_ms_min']) / 1e3, 2), r['matrix_ok']])
        out.append(table(['sample', 'kind', 'codec', 'ref', 'payload b/nnz', 'K', 'P', 'C', 'S', 'side', 'dec s', 'ok'], t))
    return '\n'.join(out)

def main():
    A = jsonl(R('results', 'idea3_a_measure.jsonl'))
    Bm = jsonl(R('results', 'idea3_b_measure.jsonl'))
    ref = ref_payload()
    print(section_a(A, ref))
    print()
    print(section_b(Bm))
    print()
    print(section_codec(ref))


def splice_notes():
    """replace the block between the markers in notes/idea3.md with the current tables"""
    p = R('notes', 'idea3.md')
    s = open(p, encoding='utf-8').read()
    a, b = '<!-- BEGIN TABLES -->', '<!-- END TABLES -->'
    i, j = s.index(a) + len(a), s.index(b)
    t = open(R('results', 'idea3_tables.md'), encoding='utf-8').read()
    s = s[:i] + '\n\n## Tables\n\n' + t + '\n' + s[j:]
    open(p, 'w', encoding='utf-8').write(s)

if __name__ == '__main__':
    if '--splice' in sys.argv:
        splice_notes()
    else:
        main()

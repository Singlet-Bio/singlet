#!/usr/bin/env python
"""Render notes/track_d.md from tools/track_d_note_template.md: every {{K<n>}} placeholder is replaced by a
table produced by tools/track_d_tables.py (so every number in a table comes straight from results/track_d_*.tsv),
{{FLOORSPLIT}} by a table computed from results/track_d_floor.tsv.

    python -P tools/track_d_render_note.py
"""
import csv, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from track_d_common import ROOT

KEYS = {
    'K1': 'Order-0 and combinatorial', 'K2': "Today's chunk payload", 'K3': 'Empty droplets: multinomial', 'K4': 'Goodness of fit',
    'K5': 'Same tests by droplet size', 'K6': 'Called cells', 'K7': 'Diminishing returns', 'K8': 'Second-pass ratio', 'K9': 'Floor estimate',
    'K10': 'Orchestrator oracle, per 1024-column chunk, entropy', 'K11': 'Orchestrator oracle, per 1024-column chunk, coders',
}


def tables():
    out = subprocess.run([sys.executable, '-P', os.path.join(HERE, 'track_d_tables.py'), 'all'], capture_output=True, text=True, cwd=ROOT).stdout
    parts = re.split(r'^### ', out, flags=re.M)[1:]
    res = {}
    for p in parts:
        head, _, body = p.partition('\n')
        for k, v in KEYS.items():
            if head.startswith(v):
                res[k] = body.strip('\n')
    return res


def floorsplit():
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'results', 'track_d_floor.tsv')), delimiter='\t'))
    hdr = ['file', 'today (bytes)', 'removable: names, permutation, pointers, colsums', 'removable: payload excess over floor', 'irreducible: called cells',
           'irreducible: empty droplets', 'irreducible: k_j side info', 'irreducible: barcode identity', 'floor total', 'ambient (empties) share of floor']
    out = ['| ' + ' | '.join(hdr) + ' |', '|' + '|'.join('---' for _ in hdr) + '|']
    for r in rows:
        t = lambda k: float(r[k])
        cells = t('cells_floor_central_bpn') * t('cells_nnz') / 8
        emp = min(t('empties_floor_loo_bpn'), t('empties_floor_heldout_mix_bpn') if r['empties_floor_heldout_mix_bpn'] not in ('', 'nan') else 1e9) * t('empties_nnz') / 8
        side = t('side_kj_cells_bytes') + t('side_kj_empties_bytes')
        bc = t('barcode_identity_bytes')
        tot = cells + emp + side + bc
        names = t('names_meta_z') + t('perm_z') + t('ptr_z') + t('colsums_z')
        excess = t('chunk_bytes') - cells - emp - side
        out.append('| ' + ' | '.join([f"{r['sample']} {r['member']}", f"{t('file_bytes'):,.0f}", f'{names:,.0f} ({100 * names / t("file_bytes"):.0f}%)',
                                     f'{excess:,.0f} ({100 * excess / t("file_bytes"):.0f}%)', f'{cells:,.0f} ({100 * cells / t("file_bytes"):.0f}%)',
                                     f'{emp:,.0f} ({100 * emp / t("file_bytes"):.0f}%)', f'{side:,.0f}', f'{bc:,.0f}', f'{tot:,.0f}', f'{100 * emp / tot:.0f}%']) + ' |')
    return '\n'.join(out)


def main():
    tpl = open(os.path.join(HERE, 'track_d_note_template.md'), encoding='utf-8').read()
    T = tables()
    T['FLOORSPLIT'] = floorsplit()
    def rep(m):
        k = m.group(1)
        if k not in T:
            return f'(table {k} not available)'
        return T[k]
    out = re.sub(r'\{\{(\w+)\}\}', rep, tpl)
    open(os.path.join(ROOT, 'notes', 'track_d.md'), 'w', encoding='utf-8').write(out)
    print('wrote notes/track_d.md', len(out), 'bytes')


if __name__ == '__main__':
    main()

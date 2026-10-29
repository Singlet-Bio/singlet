"""Parse idea2 bench logs (stdout with [idea2] lines, optional '### cfg' markers) into a flat table.
Usage: python -P tools/idea2_parse.py results/idea2_sweep_x.log [more logs]
"""
import re, sys

rows = []
for path in sys.argv[1:]:
    cfg = ''
    cur = {}
    for line in open(path, encoding='utf-8', errors='replace'):
        line = line.rstrip('\n')
        if line.startswith('### '):
            cfg = line[4:]
            continue
        m = re.match(r'== (\S+)', line)
        if m:
            cur = {'cfg': cfg, 'file': m.group(1).replace('corpus/', '').replace('_counts.1pz', '')}
            continue
        m = re.search(r'\[idea2\] payload (\d+) B \(([\d.]+) bits/nnz\).*cells (\d+) \(nnz (\d+)\) empties (\d+) \(nnz (\d+)\) S=(\d+) Tn=(\d+)', line)
        if m:
            cur.update(payload_B=int(m.group(1)), payload_bnnz=float(m.group(2)), ncells=int(m.group(3)), nnz_cells=int(m.group(4)),
                       nempt=int(m.group(5)), nnz_empt=int(m.group(6)), S=int(m.group(7)), Tn=int(m.group(8)))
            continue
        m = re.search(r'\[idea2\] cells: (.*)TOTAL ([\d.]+) bits/nnz\(cells\)', line)
        if m:
            cur['cells_bnnz'] = float(m.group(2))
            for part in m.group(1).split('|'):
                mm = re.match(r'\s*(\S+) (\d+) B \(([\d.]+) b/nnz\)', part)
                if mm: cur['c_' + mm.group(1)] = float(mm.group(3))
            continue
        m = re.search(r'\[idea2\] empties: (.*)TOTAL ([\d.]+) bits/nnz\(empties\)', line)
        if m:
            cur['empt_bnnz'] = float(m.group(2))
            for part in m.group(1).split('|'):
                mm = re.match(r'\s*(\S+) (\d+) B \(([\d.]+) b/nnz\)', part)
                if mm: cur['e_' + mm.group(1)] = float(mm.group(3))
            continue
        m = re.search(r'\[idea2\] time: order (\d+) hdr (\d+) fit (\d+) cells (\d+) empt (\d+) ms', line)
        if m:
            cur.update(t_order=int(m.group(1)), t_cells=int(m.group(4)), t_empt=int(m.group(5)))
            continue
        m = re.search(r'^\s+(idea2\S*)\s+(\d+) B\s+ratio ([\d.]+)\s+([\d.]+) bits/nnz\s+enc\s+([\d.]+) ms\s+dec\s+([\d.]+) ms \(([\d.]+) Mnnz/s\)\s+matrix=(\w+)', line)
        if m:
            cur.update(codec=m.group(1), total_B=int(m.group(2)), enc_ms=float(m.group(5)), dec_ms=float(m.group(6)), Mnnz_s=float(m.group(7)), ok=m.group(8))
            rows.append(dict(cur))
            cur = {k: cur[k] for k in ('cfg', 'file')}

cols = ['cfg', 'file', 'codec', 'ok', 'cells_bnnz', 'empt_bnnz', 'payload_bnnz', 'c_cell_top', 'c_cell_tail', 'c_cell_val', 'c_label', 'c_perm',
        'e_empt_set', 'e_empt_lab', 'e_empt_val', 'dec_ms', 'Mnnz_s', 'enc_ms', 't_order', 'payload_B', 'total_B']
print('\t'.join(cols))
for r in rows:
    print('\t'.join(str(r.get(c, '')) for c in cols))

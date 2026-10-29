"""Rebuild results/idea1_choice.tsv from the idea1_choose_*.log files (the choose runs overlapped and could overwrite each other)."""
import glob, re, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
rows = {}
for fn in sorted(glob.glob(os.path.join(R, 'idea1_choose_*.log'))):
    for ln in open(fn):
        m = re.match(r'(\S+)\s+(exon|intron)\s+best\s+(\S+)\s+([\d.]+)\s+pooled\s+([\d.]+)', ln)
        if m: rows[(m.group(1), m.group(2))] = (m.group(3), float(m.group(4)), float(m.group(5)))
with open(os.path.join(R, 'idea1_choice.tsv'), 'w') as f:
    f.write('sample\tmember\thold\n')
    for (s, mem), (h, b, p) in sorted(rows.items()): f.write(f'{s}\t{mem}\t{h}\n')
with open(os.path.join(R, 'idea1_choice_detail.tsv'), 'w') as f:
    f.write('sample\tmember\tbest\tbest_py_bits\tpooled_py_bits\n')
    for (s, mem), (h, b, p) in sorted(rows.items()): f.write(f'{s}\t{mem}\t{h}\t{b}\t{p}\n')
print(len(rows), 'choices')

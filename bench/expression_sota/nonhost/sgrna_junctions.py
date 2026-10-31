"""Leader-TRS junction (sgRNA identity) reads from STAR alignments to NC_045512.2.
usage: python -I sgrna_junctions.py viral_aligned.sam viral.trs.json OUT_PREFIX [MIN_GAP]
Evidence classes per read (primary alignments only):
  spliced : a CIGAR N gap >= MIN_GAP (default 500) whose donor (last base before the gap) is within +-12 of the end of the TRS-L core
            (leader 1..75) and whose acceptor (first base after the gap) is within [core_start-10, core_end+10] of a body TRS (ACGAAC) -> sgRNA
            of that ORF; acceptor elsewhere -> 'noncanonical' (leader donor, other acceptor).
  clipped : alignment starts at a body TRS acceptor with a leading soft clip >= 12 whose last min(clip,25) bases equal the leader tail (<=1 mismatch).
Reads without CB/UB tags count in the summary only. Output: OUT_PREFIX.percell.tsv (cb, sgrna, evidence, reads, umis), OUT_PREFIX.summary.json."""
import sys, re, json, collections

sam, trsf, out = sys.argv[1:4]
MIN_GAP = int(sys.argv[4]) if len(sys.argv) > 4 else 500
T = json.load(open(trsf))
LE = T['leader_core_end']
LS = T['leader_seq']
TB = T['trsb']
cig = re.compile(r'(\d+)([MIDNSHP=X])')


def sg_for(acc):
    for g, v in TB.items():
        if v['core_start'] - 10 <= acc <= v['core_end'] + 10:
            return g
    return None


reads = collections.Counter()
umis = collections.defaultdict(set)
nread = collections.Counter()
body_end = {}
for l in open(sam):
    if l[0] == '@':
        continue
    f = l.rstrip('\n').split('\t')
    flag = int(f[1])
    if flag & 0x100 or flag & 0x4:
        continue
    pos = int(f[3])
    ops = [(int(n), o) for n, o in cig.findall(f[5])]
    tags = {t[:2]: t[5:] for t in f[11:]}
    cb, ub = tags.get('CB', '-'), tags.get('UB', '-')
    hit = None
    p = pos
    for n, o in ops:
        if o in 'M=XD':
            p += n
        elif o == 'N':
            donor, acc = p - 1, p + n
            if n >= MIN_GAP and abs(donor - LE) <= 12:
                hit = ('spliced', sg_for(acc) or 'noncanonical')
                break
            p += n
    if hit is None and ops and ops[0][1] == 'S' and ops[0][0] >= 12:
        sg = sg_for(pos)
        if sg:
            k = min(ops[0][0], 25)
            clip = f[9][ops[0][0] - k:ops[0][0]]
            if sum(a != b for a, b in zip(clip, LS[-k:])) <= 1:
                hit = ('clipped', sg)
    nread['alignments'] += 1
    if hit:
        nread[hit[0] + ':' + hit[1]] += 1
        body_end.setdefault(hit[1], []).append(pos + sum(n for n, o in ops if o in 'MDN=X') - 1)   # last reference base of the alignment (3' end of the read in the genome)
        if cb != '-' and ub != '-':
            k = (cb, hit[1], hit[0])
            reads[k] += 1
            umis[k].add(ub)
with open(out + '.percell.tsv', 'w') as o:
    o.write('barcode\tsgrna\tevidence\treads\tumis\n')
    for k, n in sorted(reads.items()):
        o.write(f'{k[0]}\t{k[1]}\t{k[2]}\t{n}\t{len(umis[k])}\n')
cells = collections.defaultdict(set)
for (cb, sg, ev), u in umis.items():
    cells[sg].update((cb, x) for x in u)
json.dump({'min_gap': MIN_GAP, 'leader_core_end': LE, 'alignments': nread['alignments'], 'read_counts': dict(nread),
           'read_end_pos_min_median_max': {g: [min(v), sorted(v)[len(v) // 2], max(v)] for g, v in body_end.items()},
           'umis_per_sgrna_all_cells': {g: len(s) for g, s in cells.items()},
           'cells_per_sgrna': {g: len({c for c, _ in s}) for g, s in cells.items()}}, open(out + '.summary.json', 'w'), indent=1)
print(json.dumps(dict(nread)))

"""NC_045512.2 FASTA + NCBI GFF3 -> viral GTF (gene/transcript/exon per gene, 5'UTR, 3'UTR) + trs.json (TRS-L leader, TRS-B acceptors).
usage: python -I viral_ref.py genome.fna annotation.gff3 out_prefix"""
import sys, re, json
fa, gff, out = sys.argv[1:4]
seq = ''.join(l.strip() for l in open(fa) if not l.startswith('>')).upper()
chrom = next(l[1:].split()[0] for l in open(fa) if l.startswith('>'))
genes, utr = [], []
for l in open(gff):
    if l.startswith('#') or not l.strip():
        continue
    f = l.rstrip('\n').split('\t')
    if len(f) < 9:
        continue
    a = dict(kv.split('=', 1) for kv in f[8].split(';') if '=' in kv)
    s, e, st = int(f[3]), int(f[4]), f[6]
    if f[2] == 'gene':
        genes.append((a.get('Name') or a.get('gene'), s, e, st))
    elif f[2] == 'five_prime_UTR':
        utr.append(('5UTR', s, e, st))
    elif f[2] == 'three_prime_UTR':
        utr.append(('3UTR', s, e, st))
feats = genes + utr
with open(out + '.gtf', 'w') as o:
    for g, s, e, st in feats:
        at = f'gene_id "{g}"; transcript_id "{g}"; gene_name "{g}"; gene_biotype "viral";'
        for t in ('gene', 'transcript', 'exon'):
            o.write(f'{chrom}\tNCBI\t{t}\t{s}\t{e}\t.\t{st}\t.\t{at}\n')
# TRS-L: first ACGAAC core in the 5' 120 nt (leader = 1..core end).
# TRS-B: nearest upstream ACGAAC (within 200 nt of the ORF start) of each sg ORF.
core = 'ACGAAC'
L = [m.start() + 1 for m in re.finditer(core, seq[:120])]
allc = [m.start() + 1 for m in re.finditer(core, seq)]
trs = {'chrom': chrom, 'length': len(seq), 'core': core, 'leader_core_start': L[0], 'leader_core_end': L[0] + 5,
       'leader_seq': seq[:L[0] + 5], 'genes': {g: [s, e, st] for g, s, e, st in feats}, 'trsb': {}}
for g, s, e, st in genes:
    if g == 'ORF1ab' or s < 1000:
        continue
    c = [p for p in allc if p > 120 and p + 5 <= s + 5 and s - (p + 5) <= 200]
    if c:
        trs['trsb'][g] = {'core_start': c[-1], 'core_end': c[-1] + 5, 'orf_start': s}
json.dump(trs, open(out + '.trs.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in trs.items() if k != 'genes'}))

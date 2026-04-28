#!/usr/bin/env python3
"""
Debug species detection: check why some samples get 0 hits.
Tests k-mer matching manually against the DB.
"""
import sys, os, struct, re, collections

# Parse the generated k-mer DB
DB_H = "/mnt/home/debruinz/Singlet-AI/singlify/include/singlet-pileup/species_kmer_db_gen.h"

K = 21
_MASK = (1 << (2*K)) - 1
_COMP = {0:3, 1:2, 2:1, 3:0}
_BASE2BIT = {ord('A'):0,'a':0,ord('C'):1,'c':1,ord('G'):2,'g':2,ord('T'):3,'t':3}

def revcomp_kmer(kmer):
    rc=0; tmp=kmer
    for _ in range(K):
        rc=(rc<<2)|_COMP[tmp&3]; tmp>>=2
    return rc

def canonical(kmer):
    rc=revcomp_kmer(kmer); return kmer if kmer<=rc else rc

def extract_kmers_str(seq):
    km=0; v=0; out=[]
    for c in seq:
        b=_BASE2BIT.get(ord(c),-1) if isinstance(c,str) else _BASE2BIT.get(c,-1)
        if b<0: v=0; km=0; continue
        km=((km<<2)|b)&_MASK; v+=1
        if v>=K: out.append(canonical(km))
    return out

# Load DB
with open(DB_H) as f:
    content = f.read()
human_vals = set(int(v,16) for v in re.findall(r'0x([0-9a-f]{16})ULL', re.findall(r'HUMAN_KMERS\[\]\s*=\s*\{(.*?)\};', content, re.DOTALL)[0]))
mouse_vals  = set(int(v,16) for v in re.findall(r'0x([0-9a-f]{16})ULL', re.findall(r'MOUSE_KMERS\[\]\s*=\s*\{(.*?)\};', content, re.DOTALL)[0]))
print(f"DB loaded: {len(human_vals):,} human, {len(mouse_vals):,} mouse k-mers")

# Test against known sequences from R2 reads
test_reads = [
    # Mouse read from SRR34789664 (seen from decode above)
    ("mouse_SRR34789664_R2", "AGCTGCTGAAAACCACGCACAATGCAGGGATGGCAAATGC" + "A" * 50),
    # Human read from SRR32855204 (first R2 read from a human arc-gex)
    ("human_SRR32855204_R2", "GCAGAGGAAGATGAAATGCGGCAGAT" + "T" * 64),
]

for label, seq in test_reads:
    kms = extract_kmers_str(seq)
    h_hits = sum(1 for km in kms if km in human_vals)
    m_hits = sum(1 for km in kms if km in mouse_vals)
    print(f"\n{label}: {len(kms)} k-mers, human_hits={h_hits}, mouse_hits={m_hits}")

# Test against actual FASTQ reads - decode a few and check
fq_5895 = "/mnt/projects/debruinz_project/singlify_validation/1fq/SRR34789664.1fq"
fq_328  = "/mnt/projects/debruinz_project/singlify_validation/1fq/SRR32855204.1fq"
fq_885  = "/mnt/projects/debruinz_project/singlify_validation/1fq/SRR10885105.1fq"

# Add singlepress path to load .1fq
sys.path.insert(0, '/mnt/home/debruinz/Singlet-AI/singlepress')
try:
    import singlepress as sp
    for label, path in [("SRR34789664 (mouse)", fq_5895), 
                         ("SRR32855204 (human arc)", fq_328),
                         ("SRR10885105 (human 10xv2)", fq_885)]:
        try:
            # The .1fq binary format: read first block bytes for seq data
            # This requires the C++ reader; skip if singlepress can't help
            print(f"\n{label}: (need C++ reader)")
        except Exception as e:
            print(f"{label}: {e}")
except ImportError:
    print("\n(singlepress not available)")

# Instead: just check what random transcriptomic reads look like
# Use reads decoded by singlify decode above (we know R2 for SRR34789664)
actual_mouse_read = "AGCTGCTGAAAACCACGCACAATGCAGGGATGGCAAATGC" + "N" * 50
kms = extract_kmers_str(actual_mouse_read)
h_hits = sum(1 for km in kms if km in human_vals)
m_hits = sum(1 for km in kms if km in mouse_vals)
total = len(kms)
print(f"\nActual mouse read (40bp real + 50N): {total} k-mers, h={h_hits}, m={m_hits}")

# The issue: reads have Ns, reducing effective k-mer count
# Let's test the 40bp clean portion only
actual_mouse_read_clean = "AGCTGCTGAAAACCACGCACAATGCAGGGATGGCAAATGC"
kms = extract_kmers_str(actual_mouse_read_clean)
h_hits = sum(1 for km in kms if km in human_vals)
m_hits = sum(1 for km in kms if km in mouse_vals)
total = len(kms)
print(f"Clean 40bp portion: {total} k-mers, h={h_hits}, m={m_hits}")
# Actually 40bp < 21*2=42 needed for a 21-mer? No: 40-21+1=20 k-mers
for km in kms[:5]:
    in_h = km in human_vals
    in_m = km in mouse_vals
    print(f"  k-mer: {km:016x} human={in_h} mouse={in_m}")

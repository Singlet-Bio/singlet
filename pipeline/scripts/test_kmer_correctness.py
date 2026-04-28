#!/usr/bin/env python3
# Quick k-mer correctness test
K=21; _MASK=(1<<(2*K))-1; _COMP={0:3,1:2,2:1,3:0}
_BASE2BIT={ord(c):v for c,v in zip("AaCcGgTt",[0,0,1,1,2,2,3,3])}

def revcomp_kmer(kmer, k=K):
    rc=0; tmp=kmer
    for _ in range(k):
        rc=(rc<<2)|_COMP[tmp&3]; tmp>>=2
    return rc

def canonical(kmer):
    rc=revcomp_kmer(kmer)
    return kmer if kmer<=rc else rc

def extract_kmers(seq):
    km=0; v=0; out=[]
    for c in seq.encode():
        b=_BASE2BIT.get(c,-1)
        if b<0: v=0; km=0; continue
        km=((km<<2)|b)&_MASK; v+=1
        if v>=K: out.append(canonical(km))
    return out

# Verify: canonical is symmetric
for seq in ["ACTGACTGACTGACTGACTGA", "GCGCGCGCGCGCGCGCGCGCG", "ATATATATATAT"*5]:
    kms = extract_kmers(seq)
    for km in kms:
        assert canonical(km) == canonical(revcomp_kmer(km)), f"Not symmetric: {km}"
print("Canonical symmetry: OK")

# Check revcomp_kmer is correct for known sequence
seq = "ACGTACGTACGTACGTACGTA"  # 21 bases
kms = extract_kmers(seq)
km = kms[0]
rc = revcomp_kmer(km)
print(f"kmer={km:012x} revcomp={rc:012x} canonical={canonical(km):012x}")

# Check that revcomp is actually the reverse complement
def decode_kmer(kmer, k=K):
    bases = "ACGT"
    result = ""
    for _ in range(k):
        result = bases[kmer & 3] + result
        kmer >>= 2
    return result

km_seq = decode_kmer(km)
rc_seq = decode_kmer(rc)
def rev_comp(s):
    comp = str.maketrans("ACGT","TGCA")
    return s.translate(comp)[::-1]
print(f"Forward: {km_seq}")
print(f"Revcomp: {rc_seq}")
print(f"Expected: {rev_comp(km_seq)}")
print(f"Correct: {rc_seq == rev_comp(km_seq)}")

# Check that the kmer DB size makes sense
try:
    import sys
    sys.path.insert(0, ".")
    # Just check k-mer counts
    human_top_file = "include/singlet-pileup/species_kmer_db_gen.h"
    import re
    with open(human_top_file) as f:
        content = f.read()
    h_count = int(re.search(r"HUMAN_KMERS_COUNT = (\d+)", content).group(1))
    m_count = int(re.search(r"MOUSE_KMERS_COUNT = (\d+)", content).group(1))
    print(f"\nk-mer DB: human={h_count:,} mouse={m_count:,}")
    
    # Check first few k-mer values
    human_vals = re.findall(r"0x([0-9a-f]{16})ULL", content[:2000])
    print(f"First 5 human k-mers: {human_vals[:5]}")
    for v in human_vals[:5]:
        km = int(v, 16)
        print(f"  {km:016x} -> {decode_kmer(km)}")
except Exception as e:
    print(f"DB check failed: {e}")

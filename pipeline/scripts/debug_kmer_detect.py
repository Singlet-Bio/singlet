#!/usr/bin/env python3
"""Debug species k-mer detection: inspect R2 reads and check kmer hits."""
import sys, os
sys.path.insert(0, "/mnt/home/debruinz/Singlet-AI/singlepress")
import singlepress as sp
import struct

K = 21
MASK = (1 << (2*K)) - 1
COMP = {0: 3, 1: 2, 2: 1, 3: 0}

def revcomp(kmer, k=K):
    rc = 0
    tmp = kmer
    for _ in range(k):
        rc = (rc << 2) | COMP[tmp & 3]
        tmp >>= 2
    return rc

def canonical(kmer, k=K):
    rc = revcomp(kmer, k)
    return min(kmer, rc)

def numeric_kmers(seq_bytes, k=K):
    """Extract canonical k-mers from byte-numeric sequence (A=0,C=1,G=2,T=3)."""
    kmers = []
    kmer = 0
    valid = 0
    for b in seq_bytes:
        if b > 3:
            valid = 0; kmer = 0; continue
        kmer = ((kmer << 2) | b) & MASK
        valid += 1
        if valid >= k:
            kmers.append(canonical(kmer))
    return kmers

# Load the kmer DB header (parse the C++ arrays)
DB_PATH = "/mnt/home/debruinz/Singlet-AI/singlify/include/singlet-pileup/species_kmer_db_gen.h"
human_db = set()
mouse_db = set()

print("Loading kmer DB...", end="", flush=True)
current = None
with open(DB_PATH) as f:
    for line in f:
        line = line.strip()
        if "HUMAN_KMERS[]" in line:
            current = "human"
        elif "MOUSE_KMERS[]" in line:
            current = "mouse"
        elif current:
            for tok in line.replace(",","").replace("{","").replace("}","").replace(";","").split():
                try:
                    v = int(tok, 16) if tok.startswith("0x") or tok.startswith("0X") else int(tok)
                    if current == "human":
                        human_db.add(v)
                    else:
                        mouse_db.add(v)
                except:
                    pass
print(f" Human={len(human_db)} Mouse={len(mouse_db)}")

BASE = "/mnt/projects/debruinz_project/singlify_validation/1fq"

for srr, expected in [("SRR32855204", "human"), ("SRR34789664", "mouse"), ("SRR27329891", "human")]:
    fq = f"{BASE}/{srr}.1fq"
    if not os.path.exists(fq):
        print(f"{srr}: file not found"); continue

    try:
        r2_blocks = sp.read_1fq_r2_blocks(fq, max_reads=5000)
    except Exception as e:
        # fallback: try reading with r2_numeric
        try:
            r2_blocks = sp.read_1fq_reads(fq, max_reads=5000)
        except Exception as e2:
            print(f"{srr}: Cannot read — {e} / {e2}")
            continue

    total_kmers = 0
    h_hits = 0
    m_hits = 0
    n_reads = 0

    # Inspect first 5 reads
    for i, seq in enumerate(r2_blocks):
        if isinstance(seq, (bytes, bytearray)):
            seq_bytes = list(seq)
        else:
            seq_bytes = list(seq)
        
        if i < 3:
            print(f"  {srr} R2[{i}] len={len(seq_bytes)} first10={seq_bytes[:10]}")
        
        km = numeric_kmers(seq_bytes)
        total_kmers += len(km)
        for k in km:
            if k in human_db: h_hits += 1
            if k in mouse_db: m_hits += 1
        n_reads += 1
        if n_reads >= 5000:
            break

    h_rate = h_hits / max(total_kmers, 1)
    m_rate = m_hits / max(total_kmers, 1)
    call = "human" if h_rate > m_rate else "mouse" if m_rate > h_rate else "ambiguous"
    print(f"{srr} (expected={expected}): reads={n_reads} kmers={total_kmers} "
          f"h_hits={h_hits}({h_rate:.4f}) m_hits={m_hits}({m_rate:.4f}) → {call}")

#!/usr/bin/env python3
"""build_te_kmer_index.py — Build .teki binary from Dfam family consensus FASTA.

Usage:
    python build_te_kmer_index.py --fasta dfam_consensus.fa --out te_kmer_index.teki

Inputs:
    --fasta: Dfam family consensus FASTA (one record per family).
             Header format: >FAMILY_NAME  (e.g., >L1HS, >AluY, >THE1B)
    --host-txome: Optional host transcriptome FASTA (e.g., gencode.v44.transcripts.fa).
             K-mers also present in host transcripts are discarded (reduces L3 fallthrough).

Outputs:
    .teki binary: header + NUL-terminated family names + sorted k-mer entries.

Algorithm:
    1. Extract all canonical 22-mers from each family consensus.
    2. Track which family each k-mer belongs to.
    3. Keep only "family-unique" k-mers (appear in exactly 1 family).
    4. If --host-txome given, subtract host k-mers.
    5. Write sorted binary.
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections import defaultdict
from pathlib import Path


K = 22
TEKI_MAGIC   = 0x494B4554  # "TEKI"
TEKI_VERSION = 1

RC_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")


def canonical_kmer(seq: str, k: int) -> int | None:
    """Encode a k-mer to 2-bit canonical (min of fwd/revcomp). Returns None if N present."""
    if len(seq) != k:
        return None
    fwd = 0
    for c in seq:
        b = "ACGTacgt".find(c)
        if b < 0:
            return None
        b = b % 4
        fwd = (fwd << 2) | b
    # Reverse complement
    rev_seq = seq[::-1].translate(RC_TABLE)
    rev = 0
    for c in rev_seq:
        b = "ACGTacgt".find(c)
        if b < 0:
            return None
        b = b % 4
        rev = (rev << 2) | b
    return min(fwd, rev)


def read_fasta(path: str):
    """Yield (name, sequence) tuples from a FASTA file."""
    name = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq_parts)
                name = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
    if name is not None:
        yield name, "".join(seq_parts)


def extract_kmers(seq: str, k: int) -> set[int]:
    """Extract all valid canonical k-mers from a sequence."""
    kmers = set()
    for i in range(len(seq) - k + 1):
        kmer = canonical_kmer(seq[i:i+k], k)
        if kmer is not None:
            kmers.add(kmer)
    return kmers


def main():
    parser = argparse.ArgumentParser(description="Build TE family k-mer index (.teki)")
    parser.add_argument("--fasta", required=True, help="Dfam family consensus FASTA")
    parser.add_argument("--host-txome", help="Host transcriptome FASTA (k-mers to subtract)")
    parser.add_argument("--out", required=True, help="Output .teki file")
    parser.add_argument("-k", type=int, default=K, help=f"K-mer size (default: {K})")
    args = parser.parse_args()

    k = args.k

    # 1. Read all families and extract k-mers
    print(f"[build_te_kmer_index] Reading families from {args.fasta}...", file=sys.stderr)
    family_names = []
    kmer_to_families: dict[int, set[int]] = defaultdict(set)

    for name, seq in read_fasta(args.fasta):
        fam_id = len(family_names)
        family_names.append(name)
        kmers = extract_kmers(seq, k)
        for km in kmers:
            kmer_to_families[km].add(fam_id)

    print(f"  {len(family_names)} families, {len(kmer_to_families)} total distinct k-mers", file=sys.stderr)

    # 2. Keep only family-unique k-mers
    unique_entries = []
    for km, fam_ids in kmer_to_families.items():
        if len(fam_ids) == 1:
            unique_entries.append((km, next(iter(fam_ids))))

    n_multi = len(kmer_to_families) - len(unique_entries)
    print(f"  {len(unique_entries)} family-unique k-mers ({n_multi} multi-family discarded)", file=sys.stderr)

    # 3. Optionally subtract host transcriptome k-mers
    if args.host_txome:
        print(f"[build_te_kmer_index] Subtracting host k-mers from {args.host_txome}...", file=sys.stderr)
        host_kmers = set()
        for name, seq in read_fasta(args.host_txome):
            host_kmers |= extract_kmers(seq, k)
        before = len(unique_entries)
        unique_entries = [(km, fid) for km, fid in unique_entries if km not in host_kmers]
        print(f"  {before - len(unique_entries)} host-overlapping k-mers removed, {len(unique_entries)} remaining", file=sys.stderr)

    # 4. Sort by k-mer value
    unique_entries.sort(key=lambda x: x[0])

    # 5. Write binary
    print(f"[build_te_kmer_index] Writing {args.out}...", file=sys.stderr)
    with open(args.out, "wb") as f:
        # Header: 24 bytes
        f.write(struct.pack("<IIIIII",
                            TEKI_MAGIC, TEKI_VERSION, K,
                            len(family_names), len(unique_entries), 0))

        # Family names (NUL-terminated)
        for name in family_names:
            f.write(name.encode("ascii") + b"\x00")

        # K-mer entries: 12 bytes each (8B kmer_2bit + 4B family_id)
        for km, fid in unique_entries:
            f.write(struct.pack("<QI", km, fid))

    total_size = 24 + sum(len(n) + 1 for n in family_names) + len(unique_entries) * 12
    print(f"  Written {total_size:,} bytes ({total_size/1024/1024:.1f} MB)", file=sys.stderr)
    print(f"[build_te_kmer_index] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()

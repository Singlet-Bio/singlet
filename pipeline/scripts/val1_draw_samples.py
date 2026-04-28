#!/usr/bin/env python3
"""VAL1 Stratified Sample Draw Script

Draws ~200 SRR accessions from the geo-reprocess curated catalog covering
diverse protocols, species, and read-count tiers.

Output: singlify/scripts/val1_samples.csv
Columns: srr_accession, gse_series, protocol_family, species, estimated_reads, quality_tier
"""

import random
import sys
from pathlib import Path

import pandas as pd

CATALOG_PATH = "/mnt/home/debruinz/geo-reprocess/data/curated_catalog.parquet"
OUTPUT_PATH = Path(__file__).parent / "val1_samples.csv"

# Seed for reproducibility
RANDOM_SEED = 42

# ── Protocol quotas (protocol_inferred → family label, target count) ─────────
PROTOCOL_CONFIG = {
    "10xv3":        ("10x-v3",        25),
    "10xv2":        ("10x-v2",        20),
    "10xv4":        ("10x-v4",        15),
    "10xv3_5prime": ("10x-v3-5prime", 12),
    "10x_multiome": ("10x-multiome",  10),
    "scirna":       ("sci-RNA",       12),
    "dropseq":      ("Drop-seq",      15),
    "indrop":       ("InDrop",         8),
    "bd_rhapsody":  ("BD-Rhapsody",    8),
    "parse":        ("Parse",          8),
    "microwell":    ("MicroWell",      6),
    "marsseq":      ("MARS-seq",       5),
    "seqwell":      ("Seq-Well",       5),
    "splitseq":     ("SPLiT-seq",      5),
    "surecell":     ("SureCell",       5),
    "dnbelab":      ("DNBelab",        5),
}

# ── Species targets ────────────────────────────────────────────────────────────
SPECIES_MIN = {
    "Homo sapiens":  80,
    "Mus musculus":  60,
}
SPECIES_OTHER_MIN = 3   # at least 3 for each other species

# ── Read-count tiers ──────────────────────────────────────────────────────────
READ_TIER_BINS = [0, 5e6, 50e6, 200e6, float("inf")]
READ_TIER_LABELS = ["small", "medium", "large", "xlarge"]


def assign_read_tier(n: float) -> str:
    for i, (lo, hi) in enumerate(zip(READ_TIER_BINS, READ_TIER_BINS[1:])):
        if lo <= n < hi:
            return READ_TIER_LABELS[i]
    return "xlarge"


def load_catalog() -> pd.DataFrame:
    df = pd.read_parquet(CATALOG_PATH)
    # Keep only processable + SRR entries
    df = df[df["processability"].isin(["tier1_ready", "tier2_fasterq"])]
    df = df[df["run_accession"].notna() & df["run_accession"].str.startswith("SRR")]
    # One SRR per row already; drop rows without read_count
    df["read_count"] = pd.to_numeric(df["read_count"], errors="coerce")
    df = df[df["read_count"].notna() & (df["read_count"] > 0)]
    df = df[df["gse_id"].notna()]
    return df


def draw_protocol_slice(df: pd.DataFrame, rng: random.Random) -> list[dict]:
    rows = []
    used_srr = set()
    used_gsm = set()

    for proto_key, (family, quota) in PROTOCOL_CONFIG.items():
        subset = df[df["protocol_inferred"] == proto_key].copy()
        # Deduplicate by gsm_id first (one SRR per GSM)
        subset = subset.drop_duplicates(subset="gsm_id")
        subset = subset[~subset["run_accession"].isin(used_srr)]

        if len(subset) == 0:
            print(f"  WARNING: no rows for protocol {proto_key}", file=sys.stderr)
            continue

        take = min(quota, len(subset))
        sampled = subset.sample(n=take, random_state=rng.randint(0, 2**31))

        for _, r in sampled.iterrows():
            if r["run_accession"] in used_srr or r["gsm_id"] in used_gsm:
                continue
            used_srr.add(r["run_accession"])
            used_gsm.add(r["gsm_id"])
            rows.append({
                "srr_accession":   r["run_accession"],
                "gse_series":      r["gse_id"],
                "protocol_family": family,
                "species":         r["organism_normalized"],
                "estimated_reads": int(r["read_count"]),
                "quality_tier":    r["processability"],
            })

    return rows, used_srr, used_gsm


def top_up_species(df: pd.DataFrame, rows: list[dict],
                   used_srr: set, used_gsm: set, rng: random.Random) -> None:
    """Ensure species diversity targets are met."""
    current = pd.DataFrame(rows)

    def count_species(sp):
        if len(current) == 0:
            return 0
        return (current["species"] == sp).sum()

    # Human top-up
    for sp, min_n in SPECIES_MIN.items():
        deficit = min_n - count_species(sp)
        if deficit <= 0:
            continue
        candidates = df[
            (df["organism_normalized"] == sp) &
            (~df["run_accession"].isin(used_srr)) &
            (~df["gsm_id"].isin(used_gsm))
        ].drop_duplicates(subset="gsm_id")
        if len(candidates) == 0:
            continue
        take = min(deficit, len(candidates))
        sampled = candidates.sample(n=take, random_state=rng.randint(0, 2**31))
        for _, r in sampled.iterrows():
            used_srr.add(r["run_accession"])
            used_gsm.add(r["gsm_id"])
            # Assign protocol family from protocol_inferred
            family = PROTOCOL_CONFIG.get(r["protocol_inferred"],
                                         (r["protocol_inferred"], 0))[0]
            rows.append({
                "srr_accession":   r["run_accession"],
                "gse_series":      r["gse_id"],
                "protocol_family": family,
                "species":         r["organism_normalized"],
                "estimated_reads": int(r["read_count"]),
                "quality_tier":    r["processability"],
            })

    # Other species: 3 each from top non-HS/MM species
    other_species = [s for s in df["organism_normalized"].value_counts().index
                     if s not in SPECIES_MIN and df[df["organism_normalized"] == s].shape[0] >= 3]
    for sp in other_species[:8]:   # pick top 8 other species
        if count_species(sp) >= SPECIES_OTHER_MIN:
            continue
        deficit = SPECIES_OTHER_MIN - count_species(sp)
        candidates = df[
            (df["organism_normalized"] == sp) &
            (~df["run_accession"].isin(used_srr)) &
            (~df["gsm_id"].isin(used_gsm))
        ].drop_duplicates(subset="gsm_id")
        if len(candidates) == 0:
            continue
        take = min(deficit, len(candidates))
        sampled = candidates.sample(n=take, random_state=rng.randint(0, 2**31))
        for _, r in sampled.iterrows():
            used_srr.add(r["run_accession"])
            used_gsm.add(r["gsm_id"])
            family = PROTOCOL_CONFIG.get(r["protocol_inferred"],
                                         (r["protocol_inferred"], 0))[0]
            rows.append({
                "srr_accession":   r["run_accession"],
                "gse_series":      r["gse_id"],
                "protocol_family": family,
                "species":         r["organism_normalized"],
                "estimated_reads": int(r["read_count"]),
                "quality_tier":    r["processability"],
            })


def main():
    rng = random.Random(RANDOM_SEED)

    print("Loading catalog...", file=sys.stderr)
    df = load_catalog()
    print(f"Catalog: {len(df)} processable SRRs", file=sys.stderr)

    print("Drawing protocol-stratified sample...", file=sys.stderr)
    rows, used_srr, used_gsm = draw_protocol_slice(df, rng)
    print(f"After protocol draw: {len(rows)} samples", file=sys.stderr)

    print("Topping up species diversity...", file=sys.stderr)
    top_up_species(df, rows, used_srr, used_gsm, rng)
    print(f"After species top-up: {len(rows)} samples", file=sys.stderr)

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset="srr_accession")
    out["read_tier"] = out["estimated_reads"].apply(assign_read_tier)

    # Print summary
    print("\n=== Protocol distribution ===", file=sys.stderr)
    print(out["protocol_family"].value_counts().to_string(), file=sys.stderr)
    print("\n=== Species distribution ===", file=sys.stderr)
    print(out["species"].value_counts().to_string(), file=sys.stderr)
    print("\n=== Read tier distribution ===", file=sys.stderr)
    print(out["read_tier"].value_counts().to_string(), file=sys.stderr)
    print(f"\nTotal: {len(out)} samples", file=sys.stderr)

    # Write CSV (without read_tier column to keep output minimal)
    out_csv = out[["srr_accession", "gse_series", "protocol_family",
                   "species", "estimated_reads", "quality_tier"]]
    out_csv.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWritten to {OUTPUT_PATH}", file=sys.stderr)

    # Also print to stdout for quick inspection
    print(out_csv.head(5).to_csv(index=False))


if __name__ == "__main__":
    main()

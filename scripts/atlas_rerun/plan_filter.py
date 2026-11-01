#!/usr/bin/env python3
"""Plan-time exclusions: rows that cannot be a 10x gene-expression library, dropped before a byte is probed or fetched (S14).

    plan_filter.py report SAMPLES.parquet RUNS.parquet [RUNS2.parquet ...] [--organism Homo] [--out plan_excluded.tsv]

A census of 500 random multi-run human-10x samples (job 796871) found 48 without a confident protocol call; 43 of them were never 10x gene-expression: ATAC / ChIP / bulk
libraries, single-end reads (no read can hold barcode and cDNA), AB 5500xl / SOLiD / long-read instruments, and plate-based protocols. Each costs a probe (a Slurm node, an .sra
open) to be rejected by the gate. The rules use only the run tables (ENA / SRA metadata), which the plan already has:

  non_rna_assay:<strategy>      every run's library_strategy is a non-RNA assay (ATAC-seq, ChIP-Seq, Bisulfite-Seq, Hi-C, ...); a GSM that also has an RNA-Seq run stays (multiome);
  single_end                    every run is SINGLE layout with a median spot length < 80 nt (longer single reads can hold barcode and cDNA concatenated);
  plate_or_longread_instrument  every run was sequenced on AB 5500xl / SOLiD / Ion Torrent / 454 / Helicos / PacBio / Nanopore;
  unrecoverable:<reason>        the GSM is listed in unrecoverable.tsv (its deposited reads cannot be processed, with the evidence).

`apply(u, runs, unrecoverable)` -> (kept u, excluded DataFrame gsm, gse, reason). Needs pandas (it runs where the universe tables are).
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNRECOVERABLE = os.path.join(HERE, "unrecoverable.tsv")

NON_RNA = {"ATAC-seq", "ChIP-Seq", "Bisulfite-Seq", "MNase-Seq", "DNase-Hypersensitivity", "Hi-C", "CLIP-Seq", "MeDIP-Seq", "MRE-Seq", "FAIRE-seq", "ChIA-PET"}
# not AMPLICON / Targeted-Capture / WGS ...: 150 catalogue-10x GSMs carry those strategies and were processed as scRNA (submitters mislabel 10x libraries)
SINGLE_MAX_SPOT = 80   # a SINGLE-layout run only rules itself out below this median spot length (bases): 26-28 nt of barcode+UMI plus a usable cDNA read need more. Catalogue-10x SINGLE rows by
#                        spot length: <= 80 nt: 1512 rejected / 78 done, > 80 nt: 276 rejected / 389 done (SRA stores many 10x runs as one concatenated read)
BAD_INSTRUMENTS = ("AB 5500", "AB SOLiD", "SOLiD", "Ion Torrent", "Ion S5", "Ion Proton", "454 GS", "Helicos", "PacBio", "Sequel", "Oxford Nanopore", "MinION", "GridION", "PromethION")


def read_unrecoverable(path=UNRECOVERABLE):
    """-> {gsm: reason} from unrecoverable.tsv (gsm<TAB>reason<TAB>evidence; # comments)."""
    out = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) >= 2 and f[0] != "gsm":
                    out[f[0]] = f[1] + (" (" + f[2] + ")" if len(f) > 2 and f[2] else "")
    except OSError:
        pass
    return out


def reasons(runs, unrecoverable=None):
    """runs: DataFrame with gsm, library_strategy, library_layout, instrument_model (one row per run) -> {gsm: reason} for the GSMs to exclude."""
    out = {}
    runs = runs.copy()
    for col in ("library_strategy", "library_layout", "instrument_model"):   # a run table without the column excludes nothing by that rule
        if col not in runs:
            runs[col] = None
    for gsm, x in runs.groupby("gsm"):
        strat = [str(v) for v in x.library_strategy.dropna()]
        if strat and all(v in NON_RNA for v in strat):
            out[gsm] = "non_rna_assay:" + sorted(set(strat))[0]
            continue
        lay = [str(v).upper() for v in x.library_layout.dropna()]
        spot = (x.base_count / x.read_count.where(x.read_count > 0)).dropna() if "base_count" in x and "read_count" in x else []
        if lay and all(v == "SINGLE" for v in lay) and len(spot) and float(spot.median()) < SINGLE_MAX_SPOT:
            out[gsm] = "single_end"
            continue
        ins = [str(v) for v in x.instrument_model.dropna()]
        if ins and all(any(b.lower() in v.lower() for b in BAD_INSTRUMENTS) for v in ins):
            out[gsm] = "plate_or_longread_instrument:" + sorted(set(ins))[0]
    for gsm, why in (unrecoverable or {}).items():
        out.setdefault(gsm, "unrecoverable:" + why)
    return out


def apply(u, runs, unrecoverable=None, gsm_col="gsm_id", gse_col="gse_id"):
    """u: the universe DataFrame (gsm_id, gse_id, ...), runs: the run tables (gsm, ...). -> (u without the excluded GSMs, DataFrame(gsm, gse, reason))."""
    import pandas as pd
    why = reasons(runs[runs.gsm.isin(set(u[gsm_col]))], unrecoverable)
    why = {g: r for g, r in why.items() if g in set(u[gsm_col])}
    ex = u[u[gsm_col].isin(why)][[gsm_col, gse_col]].rename(columns={gsm_col: "gsm", gse_col: "gse"})
    ex["reason"] = ex.gsm.map(why)
    return u[~u[gsm_col].isin(why)].copy(), ex.sort_values("gsm").reset_index(drop=True) if len(ex) else pd.DataFrame(columns=["gsm", "gse", "reason"])


def write_excluded(path, ex):
    ex.to_csv(path, sep="\t", index=False)


def summary(ex, total):
    by = ex.reason.str.split(":").str[0].value_counts().to_dict() if len(ex) else {}
    return {"excluded": int(len(ex)), "of": int(total), "by_reason": by}


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("report",))
    ap.add_argument("samples")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--organism", default="Homo")
    ap.add_argument("--out")
    a = ap.parse_args()
    s = pd.read_parquet(a.samples)
    runs = pd.concat([pd.read_parquet(p) for p in a.runs], ignore_index=True).drop_duplicates("run_accession")
    if "gsm" not in runs:
        runs["gsm"] = None
    runs["gsm"] = runs["gsm"].fillna(runs["sample_alias"].where(runs["sample_alias"].astype(str).str.startswith("GSM"), runs["sample_accession"]))
    unrec = read_unrecoverable()
    pi = s.protocol_inferred.fillna("")
    human = s.organism.fillna("").str.contains(a.organism)
    sets = {
        "human, protocol_inferred 10x* (census universe, all samples)": s[human & pi.str.startswith("10x")],
        "human, catalogue 10x GEX chemistry (10xv2/10xv3/10xv3_5prime/citeseq)": s[human & pi.isin(["10xv2", "10xv3", "10xv3_5prime", "citeseq"])],
    }
    nruns = runs.groupby("gsm").size()
    for name, u in sets.items():
        for label, uu in (("all samples", u), ("multi-run samples", u[u.gsm_id.map(nruns).fillna(0) >= 2])):
            kept, ex = apply(uu, runs, unrec)
            print("%-72s %-18s %s" % (name, label, summary(ex, len(uu))))
            if len(ex):   # false-positive check against what the alpha pipeline did with the same rows
                st = ex.assign(r=ex.reason.str.split(":").str[0]).merge(s[["gsm_id", "processing_status"]], left_on="gsm", right_on="gsm_id").groupby(["r", "processing_status"]).size()
                print("    catalogue status of the excluded:", {"%s/%s" % k: int(v) for k, v in st.items()})
            if a.out and label == "all samples" and name.startswith("human, catalogue"):
                write_excluded(a.out, ex)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Pick corpus libraries whose protocol GEO states, for the protocol-detection panel.

Reads the Beta corpus tables (runs.parquet, samples.parquet), keeps one gene-expression run per study for each stated assay below, drops every
study or run named in the exclusion file (accessions that detection must never see), and writes panel_corpus.tsv in the layout of
panel_design.tsv. The stated assay is the GEO text, so these rows are weaker truth than the by-design rows.
usage: panel_pick.py CORPUS_OUT_DIR EXCLUDE_FILE OUT.tsv
"""
import sys
import pandas as pd

PICK = {  # stated assay -> (truth class, libraries)
    "seqwell": ("dropseq", 6), "dropseq": ("dropseq", 6), "dnbelab_c4": ("dnbelab-c4", 6), "microwell_seq": ("microwell-seq", 5),
    "indrop": ("indrop", 5), "bd_rhapsody": ("bd-rhapsody", 5), "10x_flex": ("flex", 4), "singleron": ("singleron", 5),
    "pipseq": ("pipseq", 4), "visium_hd": ("visium-hd", 3), "stereoseq": ("stereo-seq", 3), "curio": ("slide-seq", 3),
    "bulk": ("bulk-or-plate", 3), "split_pool": ("split-pool", 2), "10x_3p_v3": ("10x-3p", 2), "10x_5p": ("10x-5p", 2),
}
LIBTYPE = {  # GEO-stated library type -> (truth class, libraries), any assay
    "feature": ("adt", 4), "vdj": ("vdj", 4), "atac": ("atac", 4),
}


def main(out_dir, excl_file, dest, seed=20261109):
    excl = {x.strip() for x in open(excl_file) if x.strip() and not x.startswith("#")}
    r = pd.read_parquet(out_dir + "/runs.parquet")
    s = pd.read_parquet(out_dir + "/samples.parquet", columns=["sample", "assay", "organism"])
    m = r.merge(s, on="sample", how="inner")
    keys = [c for c in ("run_accession", "sample", "study", "gse", "gsm", "study_accession", "secondary_study_accession") if c in m.columns]
    bad = m[keys].astype(str).isin(excl).any(axis=1)
    m = m[~bad & m.fastq_urls.fillna("").str.count(";").eq(1) & (m.read_count.fillna(0) >= 1_000_000)]
    out = []
    for lt, (truth, k) in LIBTYPE.items():
        a = m[m.lib_type == lt].sample(frac=1.0, random_state=seed).drop_duplicates("study").head(k)
        for _, x in a.iterrows():
            u = x.fastq_urls.split(";")
            out.append([f"{lt}_{x.run_accession}", "corpus (GEO-stated)", truth, x.run_accession, u[0], u[1], f"{x.study} ({x['sample']})", f"stated {lt} library; {x.assay}; {x.organism}"])
    m = m[m.lib_type == "gex"]
    for assay, (truth, k) in PICK.items():
        a = m[m.assay == assay].sample(frac=1.0, random_state=seed).drop_duplicates("study").head(k)
        for _, x in a.iterrows():
            u = x.fastq_urls.split(";")
            out.append([f"{assay}_{x.run_accession}", "corpus (GEO-stated)", truth, x.run_accession, u[0], u[1], f"{x.study} ({x['sample']})", f"stated {assay}; {x.organism}"])
    pd.DataFrame(out, columns=["id", "group", "truth", "run", "r1_url", "r2_url", "source", "note"]).to_csv(dest, sep="\t", index=False)
    print(len(out), "libraries", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:4])

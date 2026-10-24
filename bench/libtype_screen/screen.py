# SPDX-License-Identifier: MIT
"""Screen atlas samples for non-gene-expression libraries processed as RNA.

Inputs (this folder): d1_qc.parquet (D1 sample_qc + gsm), geo_samples.parquet (GEO family SOFT, every sample of each
atlas series), members.parquet (per-sample member sizes from each bundle's ZIP central directory).
Output: libtype_suspects.tsv, libtype_by_study.tsv, libtype_all_scored.parquet.
"""

import difflib
import os
import re

import numpy as np
import pandas as pd

# working directory for caches and outputs (default: next to this script)
HERE = os.environ.get("LIBTYPE_WORK", os.path.dirname(os.path.abspath(__file__)))

# Library-type vocabulary. Tokens are matched on word boundaries where "word" excludes letters only, so
# "S133_TCR" and "TCR-seq" match TCR, but "TCRb-/-" or "anti-TCR" are handled by the negative patterns below.
CLASSES = {
    "VDJ": [
        r"V\(D\)J",
        r"sc(?:TCR|BCR)(?:[-_ ]?seq)?",
        r"(?:TCR|BCR)seq",
        r"VDJ(?:[-_ ]?[TB])?",
        r"TCR(?:[-_ ]?seq)?",
        r"BCR(?:[-_ ]?seq)?",
        r"TCR[ -]?(?:alpha/beta|ab)",
        r"immune[ -]repertoire",
        r"(?:TCR|BCR|Ig) repertoire",
        r"clonotypes?",
        r"(?:TRA|TRB)[ /]+(?:TRA|TRB)",
        r"Ig(?:[ -]?seq)",
        r"IGH",
        r"TRUST4",
    ],
    "HTO": [
        r"HTOs?",
        r"hashtags?",
        r"hashtag[ -]oligo\w*",
        r"cell[ -]hashing",
        r"hashing",
        r"hashed[ -]library",
        r"sample[ -]tags?(?: library)?",
        r"SMK",
    ],
    "ADT": [
        r"ADTs?",
        r"CITE[ -]?seq[ -_]?(?:ADT|protein|antibody|surface)",
        r"antibody[ -]derived(?: tags?)?",
        r"antibody[ -]capture",
        r"antibody[ -]oligo\w*",
        r"TotalSeq",
        r"AbSeq",
        r"surface[ -]protein(?: library)?",
        r"protein[ -](?:library|expression library|tags?)",
        r"CSP",
        r"ADT[ -]?seq",
        r"CITE",
        r"Ab[ -]?(?:library|lib)",
        r"feature[ -]barcod\w*",
    ],
    "CMO": [
        r"CMOs?",
        r"CellPlex",
        r"multiplexing[ -]capture",
        r"MULTI[ -]seq(?: barcodes?)?",
        r"LMOs?",
        r"cell[ -]multiplexing oligo\w*",
    ],
    "GUIDE": [
        r"sgRNAs?",
        r"gRNAs?",
        r"guides?(?:[ -]RNA)?",
        r"GDO",
        r"CRISPR[ -](?:guide|capture|library)",
        r"guide[ -]capture",
        r"perturbation[ -]barcodes?",
        r"dial[ -]?out",
    ],
    "ATAC": [r"ATAC", r"scATAC", r"chromatin accessibility"],
}
GEX = [
    r"GEX",
    r"GE",
    r"mRNA",
    r"RNA",
    r"gene[ -]expression",
    r"transcriptome",
    r"transcriptomics?",
    r"WTA",
    r"scRNA(?:[ -]?seq)?",
    r"snRNA(?:[ -]?seq)?",
    r"RNA[ -]?seq",
    r"5'?GEX",
    r"3'?GEX",
    r"Expression",
    r"GEXGDO",
    r"gene[ _-]?exp",
    r"GEX\w*",
]
# contexts where a VDJ word does not describe the library
NEG_VDJ = re.compile(
    r"(?i)(anti[ -]?(?:TCR|CD3)|TCR[ -]?(?:transgenic|tg\b|KO\b|-/-|deficient|knock|stimulat|activat|signal|engag|"
    r"[αβab]?[ -]?(?:KO|-/-))|OT-?I+|(?:TCR|BCR)[ -]?(?:triggering|crosslink|ligation)|TCRb?-?/-?\b)"
)


def rx(words):
    return re.compile(r"(?i)(?<![A-Za-z])(?:" + "|".join(words) + r")(?![A-Za-z])")


CLASS_RX = {k: rx(v) for k, v in CLASSES.items()}
GEX_RX = rx(GEX)
SUPP_VDJ = re.compile(
    r"(?i)(contig_annotations|clonotypes?\.|airr_rearrangement|consensus_annotations|all_contig|vdj|_tcr[_.]|_bcr[_.]|filtered_contig|vloupe)"
)
SUPP_FB = {
    "HTO": re.compile(r"(?i)(hto|hashtag)"),
    "ADT": re.compile(r"(?i)(adt|cite|antibody|protein)"),
    "CMO": re.compile(r"(?i)(cmo|cellplex|multiplexing_capture)"),
    "GUIDE": re.compile(r"(?i)(sgrna|grna|guide|gdo|crispr_analysis|protospacer)"),
    "ATAC": re.compile(r"(?i)(atac|fragments\.tsv|peaks?\.bed|peak_bc_matrix)"),
}
SUPP_GEX = re.compile(
    r"(?i)(matrix\.mtx|barcodes\.tsv|features\.tsv|genes\.tsv|feature_bc_matrix|\.h5ad|\.rds|\.h5\b|umi\.txt|counts?\.(?:txt|csv|tsv)|gex|_rna[_.]|mrna)"
)
NONRNA_STRATEGY = {
    "ATAC-seq": "ATAC",
    "ChIP-Seq": "OTHER_DNA",
    "WGS": "OTHER_DNA",
    "Bisulfite-Seq": "OTHER_DNA",
    "WXS": "OTHER_DNA",
    "AMPLICON": "AMPLICON",
    "Targeted-Capture": "TARGETED",
    "MNase-Seq": "OTHER_DNA",
    "DNase-Hypersensitivity": "OTHER_DNA",
    "Hi-C": "OTHER_DNA",
    "ChIA-PET": "OTHER_DNA",
    "Tn-Seq": "OTHER_DNA",
    "OTHER": None,
    "RNA-Seq": None,
}


def hits(text):
    out = {k: sorted({m.group(0) for m in r.finditer(text)}) for k, r in CLASS_RX.items()}
    if out["VDJ"] and NEG_VDJ.search(text):
        # drop VDJ words that only occur inside a negative context
        stripped = NEG_VDJ.sub(" ", text)
        out["VDJ"] = sorted({m.group(0) for m in CLASS_RX["VDJ"].finditer(stripped)})
    return {k: v for k, v in out.items() if v}


def norm_stem(title):
    t = re.sub(r"(?i)CITE[ -]?seq", " ", title)
    for r in list(CLASS_RX.values()) + [GEX_RX]:
        t = r.sub(" ", t)
    t = re.sub(
        r"(?i)\b(library|libraries|lib|seq|sample|data|counts?|derived|cDNA|oligo|tags?|capture|reads?|5'?|3'?)\b",
        " ",
        t,
    )
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def classify(row):
    """Text evidence for one GEO sample record -> (class, strength, evidence list, gex_mentioned)."""
    ev, votes = [], {}

    def add(field, h, w):
        for k, words in h.items():
            votes[k] = votes.get(k, 0) + w
            ev.append(f"{field}:{k}[{','.join(words)}]")

    title = row.get("title") or ""
    th = hits(title)
    t_gex = bool(GEX_RX.search(title))
    add("title", th, 3)
    strat = row.get("library_strategy") or ""
    if NONRNA_STRATEGY.get(strat):
        votes[NONRNA_STRATEGY[strat]] = votes.get(NONRNA_STRATEGY[strat], 0) + 3
        ev.append(f"library_strategy:{strat}")
    sel = " ".join(
        [
            row.get("library_selection") or "",
            row.get("library_source") or "",
            row.get("molecule") or "",
        ]
    )
    add("library_meta", hits(sel), 2)
    desc = row.get("description") or ""
    d0 = desc.split(" | ")
    # description lines that are file names are weak (they are often shared with the GEX sibling)
    d_text = " | ".join(
        x for x in d0 if not re.search(r"\.(gz|tsv|csv|mtx|h5|txt|rds|h5ad|zip|tar)\b", x)
    )
    dh = hits(d_text)
    d_gex = bool(GEX_RX.search(d_text))
    add("description", dh, 2)
    ch = row.get("characteristics") or ""
    ch_lib = " | ".join(
        x
        for x in ch.split(" | ")
        if re.search(
            r"(?i)(librar|assay|data type|seq type|modality|sequencing|molecule|chemistry|readout)",
            x.split(":")[0],
        )
    )
    add("characteristics", hits(ch_lib), 2)
    supp = row.get("supp") or ""
    sh = {}
    if SUPP_VDJ.search(supp):
        sh["VDJ"] = [SUPP_VDJ.search(supp).group(0)]
    for k, r in SUPP_FB.items():
        m = r.search(supp)
        if m:
            sh[k] = [m.group(0)]
    add("supp", sh, 1)
    # the library word nearest the end of the title wins ("Hashtag-ADT" is the ADT library of the Hashtag experiment)
    t_cls = None
    if th:
        last = {
            k: max(m.end() for m in CLASS_RX[k].finditer(title))
            for k in th
            if CLASS_RX[k].search(title)
        }
        t_cls = max(last, key=last.get) if last else max(th, key=lambda k: len(th[k]))
    t_join = bool(t_cls and t_gex and re.search(r"(?i)(\+|&|\band\b|\bwith\b|/)", title))
    if not votes:
        return None, 0, "", t_gex, d_gex, None, False
    cls = max(votes, key=votes.get)
    return cls, votes[cls], "; ".join(ev), t_gex, d_gex, t_cls, t_join


def qc_pattern(r):
    """QC fingerprint from the atlas numbers.

    'gex': looks like a gene-expression library (>= 200 genes per cell, >= 10% exonic reads, >= 50% mapped; 5th percentile of
           exonic fraction over samples with no library words and >= 300 genes is 0.075).
    'fb':  feature-barcode-like: reads do not land in transcripts (mapping < 20% and < 30 genes, or < 5% exonic and < 50 genes).
    'vdj': maps, but few genes (< 300) without looking like gene expression (amplicon of TR/IG constant + V regions; the
           constant-region exons make 10-40% of reads exonic, so exonic fraction alone does not separate it).
    'atac': many 'genes' but < 7% exonic reads (genomic fragments spread over gene bodies).
    '':    none of these (intermediate, or no QC numbers).
    """
    m, g, ex = r.mapping_rate, r.median_genes, r.exonic_fraction
    if pd.isna(m) or pd.isna(g):
        return ""
    ex = 1.0 if pd.isna(ex) else ex
    if g >= 200 and ex >= 0.1 and m >= 0.5:
        return "gex"
    if (m < 0.2 and g < 30) or (ex < 0.05 and g < 50):
        return "fb"
    if m >= 0.35 and g < 300 and not (m >= 0.5 and g >= 300 and ex < 0.07):
        return "vdj"
    if m >= 0.5 and g >= 300 and ex < 0.07:
        return "atac"
    return ""


def main():
    qc = pd.read_parquet(os.path.join(HERE, "d1_qc.parquet"))
    geo = pd.read_parquet(os.path.join(HERE, "geo_samples.parquet"))
    mem = (
        pd.read_parquet(os.path.join(HERE, "members.parquet"))
        if os.path.exists(os.path.join(HERE, "members.parquet"))
        else None
    )

    # per-sample bytes: vdj_gene_usage vs exon counts
    if mem is not None:
        mem["kind"] = np.select(
            [
                mem.member.str.contains("vdj", case=False),
                mem.member.str.contains("exon", case=False),
                mem.member.str.contains("intron", case=False),
            ],
            ["vdj", "exon", "intron"],
            "other",
        )
        mb = mem.pivot_table(
            index="gsm_id", columns="kind", values="bytes", aggfunc="sum", fill_value=0
        )
        for c in ("vdj", "exon", "intron"):
            if c not in mb:
                mb[c] = 0
        mb["vdj_exon_ratio"] = mb["vdj"] / mb["exon"].replace(0, np.nan)
        qc = qc.merge(
            mb[["vdj", "exon", "vdj_exon_ratio"]].rename(
                columns={"vdj": "vdj_bytes", "exon": "exon_bytes"}
            ),
            left_on="gsm_id",
            right_index=True,
            how="left",
        )
    else:
        qc["vdj_bytes"] = qc["exon_bytes"] = qc["vdj_exon_ratio"] = np.nan

    geo = geo.drop_duplicates("gsm_id")
    res = geo.apply(
        lambda r: pd.Series(
            classify(r),
            index=[
                "text_class",
                "text_score",
                "text_evidence",
                "gex_word",
                "desc_gex",
                "title_class",
                "title_join",
            ],
        ),
        axis=1,
    )
    geo = pd.concat([geo, res], axis=1)
    geo["stem"] = geo.title.fillna("").map(norm_stem)
    geo["in_atlas"] = geo.gsm_id.isin(qc.gsm_id)

    df = qc.merge(geo.drop(columns=["gse_id"]), on="gsm_id", how="left")
    df["qc_pattern"] = df.apply(qc_pattern, axis=1)

    # series context: do sibling samples in the same series carry a GEX word (paired-library design)?
    sib = geo.groupby("gse_id").agg(
        n_geo=("gsm_id", "size"),
        n_gex_word=("gex_word", "sum"),
        n_classed=("text_class", lambda s: s.notna().sum()),
    )
    df = df.merge(sib, on="gse_id", how="left")

    # matching mRNA GSM: same series, same title stem, GEX word or no library class
    gex_like = geo[geo.title_class.isna() | geo.gex_word.astype(bool)]
    gex_like = gex_like[gex_like.stem != ""]
    by_key = {}
    for r in gex_like.itertuples():
        by_key.setdefault((r.gse_id, r.stem), []).append(r.gsm_id)

    gw = geo[geo.gex_word.astype(bool)]
    gex_word_titles = {}
    for r in gw.itertuples():
        gex_word_titles.setdefault(r.gse_id, []).append((r.gsm_id, r.title or ""))

    def match(r):
        if not isinstance(r.stem, str) or not r.stem:
            return ""
        c = [g for g in by_key.get((r.gse_id, r.stem), []) if g != r.gsm_id]
        if not c:
            # fallback: the closest GEX-worded title in the series, if clearly closest (titles that differ in one word)
            cand = gex_word_titles.get(r.gse_id, [])
            sc = sorted(
                (
                    (difflib.SequenceMatcher(None, r.title.lower(), t.lower()).ratio(), g)
                    for g, t in cand
                    if g != r.gsm_id
                ),
                reverse=True,
            )
            if sc and sc[0][0] >= 0.75 and (len(sc) == 1 or sc[0][0] - sc[1][0] >= 0.05):
                return sc[0][1]
            return ""
        n = int(r.gsm_id[3:])
        return min(c, key=lambda g: abs(int(g[3:]) - n))

    # decide
    FB = {"HTO", "ADT", "CMO", "GUIDE"}

    def decide(r):
        tc = r.text_class if isinstance(r.text_class, str) else None
        ts, qp = (r.text_score if not pd.isna(r.text_score) else 0), r.qc_pattern
        t_cls = r.title_class if isinstance(r.title_class, str) else None
        t_gex = bool(r.gex_word) if not pd.isna(r.gex_word) else False
        d_gex = bool(r.desc_gex) if not pd.isna(r.desc_gex) else False

        def agrees(c):
            return (
                (qp == "fb" and c in FB | {"VDJ", "ATAC", "OTHER_DNA"})
                or (qp == "vdj" and c == "VDJ")
                or (qp == "atac" and c in {"ATAC", "OTHER_DNA"})
            )

        lib = conf = None
        if tc in ("OTHER_DNA", "AMPLICON", "TARGETED", "ATAC") and "library_strategy" in (
            r.text_evidence or ""
        ):
            lib, conf = tc, "high" if agrees(tc) else "medium"
        elif t_cls and not t_gex:
            # the sample title names a non-GEX library and no GEX library
            lib, conf = t_cls, "high" if agrees(t_cls) else "medium"
        elif t_cls and t_gex:
            if agrees(t_cls):
                lib, conf = t_cls, "medium"
            elif r.title_join:
                lib, conf = t_cls + "+GEX", "low"  # one GSM, several libraries ("GEX + ADT")
        elif tc and ts >= 2 and not t_gex:
            # description / characteristics / library fields name a non-GEX library
            if agrees(tc):
                lib, conf = tc, "high" if not d_gex else "medium"
            elif d_gex:
                lib, conf = tc + "+GEX", "low"
            else:
                lib, conf = tc, "low"
        elif tc and ts == 1 and agrees(tc) and not t_gex:
            lib, conf = tc, "low"  # only a supplementary file name, but the QC pattern agrees
        elif qp == "fb" and (r.n_classed or 0) > 0 and not t_gex:
            lib, conf = "UNSPECIFIED_FB", "low"  # QC-only, in a series that names non-GEX libraries
        if lib and qp == "gex":
            conf = "cleared"  # the words say otherwise, but the numbers are those of a gene-expression library
        return pd.Series({"lib_class": lib, "confidence": conf})

    df = pd.concat([df, df.apply(decide, axis=1)], axis=1)
    flagged = df.lib_class.notna()
    df["mrna_gsm"] = ""
    df.loc[flagged, "mrna_gsm"] = df[flagged].apply(match, axis=1)
    df["mrna_in_atlas"] = df.mrna_gsm.isin(qc.gsm_id) & (df.mrna_gsm != "")
    df.to_parquet(os.path.join(HERE, "libtype_all_scored.parquet"))
    sus = df[df.lib_class.notna()].copy()

    def rec(r):
        base = r.lib_class.replace("?", "")
        if r.confidence == "cleared":
            return "keep (QC is gene-expression-like)"
        if base.endswith("+GEX"):
            return (
                "keep_review: GSM names several libraries; check SRA runs and drop the non-GEX runs"
            )
        if r.mrna_in_atlas:
            return "exclude (mRNA library already in atlas)"
        if r.mrna_gsm:
            return "replace with mRNA library " + r.mrna_gsm
        return "exclude"

    sus["recommendation"] = sus.apply(rec, axis=1)
    cols = [
        "gse_id",
        "gsm_id",
        "lib_class",
        "confidence",
        "recommendation",
        "mrna_gsm",
        "mrna_in_atlas",
        "title",
        "library_strategy",
        "qc_pattern",
        "median_genes",
        "median_umi",
        "mapping_rate",
        "exonic_fraction",
        "n_cells_called",
        "vdj_exon_ratio",
        "text_evidence",
        "protocol",
    ]
    order = {"high": 0, "medium": 1, "low": 2, "cleared": 3}
    sus = (
        sus.sort_values(["gse_id", "gsm_id"])
        .assign(_o=sus.confidence.map(order))
        .sort_values(["_o", "gse_id", "gsm_id"])
        .drop(columns="_o")
    )
    sus[cols].to_csv(os.path.join(HERE, "libtype_suspects.tsv"), sep="\t", index=False)
    n_atlas = df.groupby("gse_id").size()
    fl = sus[sus.confidence != "cleared"]
    by = fl.groupby("gse_id").agg(
        n_flagged=("gsm_id", "size"),
        n_high=("confidence", lambda s: (s == "high").sum()),
        n_medium=("confidence", lambda s: (s == "medium").sum()),
        n_low=("confidence", lambda s: (s == "low").sum()),
        classes=("lib_class", lambda s: ",".join(sorted(set(s)))),
        n_mrna_in_atlas=("mrna_in_atlas", "sum"),
        n_replace=("recommendation", lambda s: s.str.startswith("replace").sum()),
    )
    by.insert(0, "n_atlas_samples", n_atlas.reindex(by.index))
    by["n_cleared"] = (
        sus[sus.confidence == "cleared"]
        .groupby("gse_id")
        .size()
        .reindex(by.index)
        .fillna(0)
        .astype(int)
    )
    by.sort_values(["n_high", "n_flagged"], ascending=False).to_csv(
        os.path.join(HERE, "libtype_by_study.tsv"), sep="	"
    )
    print(sus.groupby(["lib_class", "confidence"]).size().unstack(fill_value=0))
    for k in ("high", "medium", "low", "cleared"):
        x = sus[sus.confidence == k]
        print(k, "samples", len(x), "studies", x.gse_id.nunique())


if __name__ == "__main__":
    main()

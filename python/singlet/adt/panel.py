# SPDX-License-Identifier: MIT
"""Antibody panels: the TotalSeq catalog, study feature references, and panel identification from the reads.

A panel is a list of features ``(id, name, sequence, kind, barcode_id, ensembl, source)``, where ``kind`` is ``antibody``,
``isotype`` or ``hashtag``. Where it comes from, in order of preference:

1. ``study``: a feature reference of the study itself (Cell Ranger ``feature_reference.csv`` or a CITE-seq-Count tag list,
   from GEO supplementary files or the paper), curated under ``bench/adt/panels/<GSE>.csv``.
2. ``catalog``: identified from the reads. Every 15-mer at the library's feature offset in a sample of read pairs whose
   cell barcode is one of the GEX sample's called cells is matched (exact or Hamming 1) to the vendor catalog
   (``data/totalseq_catalog.tsv``). Catalog barcodes with clear support are the panel. This names antibodies only by
   sequence identity with a vendor list; it never infers a name from anything else.

The panel is **unresolved** (the library is flagged, not counted under invented names) when the reads at the best offset
match no catalog sequence, or when abundant sequences that are not in the catalog carry more than
``UNEXPLAINED_MAX`` of the feature-like reads: then a study reference is needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import seq as S

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "data", "totalseq_catalog.tsv")
BC_LEN = 15
OFFSETS = tuple(range(0, 26))
MIN_OFFSET_FRACTION = 0.10  # reads matching the catalog exactly at the best offset
MIN_READS = 50  # support for a catalog barcode in the sample (called-cell reads)
MIN_SHARE = 2e-4  # ... and at least this share of the matched reads
UNEXPLAINED_MAX = 0.05
ABUNDANT_SHARE = 1e-3  # an unmatched sequence this abundant (share of feature-like reads) counts as unexplained


def load_catalog(path: str = CATALOG_PATH) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


@dataclass
class Panel:
    features: pd.DataFrame  # id, name, sequence, kind, barcode_id, ensembl, source (+ reads in the discovery sample)
    source: str  # study | catalog
    offset: int
    status: str  # ok | incomplete | unresolved
    notes: Dict[str, object]

    @property
    def sequences(self) -> List[str]:
        return list(self.features.sequence)


def read_feature_reference(path: str) -> pd.DataFrame:
    """Cell Ranger feature reference (id,name,read,pattern,sequence,feature_type) or a two-column name,sequence list."""
    df = pd.read_csv(path, sep=None, engine="python", dtype=str, comment="#")
    cols = {c.lower().strip(): c for c in df.columns}
    if "sequence" in cols:
        out = pd.DataFrame({"id": df[cols.get("id", cols["sequence"])], "name": df[cols.get("name", cols.get("id"))],
                            "sequence": df[cols["sequence"]].str.upper().str.strip()})
        if "pattern" in cols:
            out["pattern"] = df[cols["pattern"]]
        if "feature_type" in cols:
            out["feature_type"] = df[cols["feature_type"]]
    else:  # headerless name,sequence
        df = pd.read_csv(path, sep=None, engine="python", dtype=str, header=None)
        out = pd.DataFrame({"id": df[0], "name": df[0], "sequence": df[1].str.upper().str.strip()})
    out = out[out.sequence.str.fullmatch(r"[ACGT]+")].reset_index(drop=True)
    return out


def kind_of(name: str) -> str:
    n = name.lower()
    if "hashtag" in n or n.startswith("hto") or "hto" in n.split("_"):
        return "hashtag"
    if "isotype" in n or ("igg" in n and "ctrl" in n) or "negctr" in n or "neg_ctr" in n or "negative control" in n:
        return "isotype"
    return "antibody"


def annotate_from_catalog(ref: pd.DataFrame, cat: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Add catalog facts (barcode id, kind, Ensembl gene) to a study reference by sequence identity."""
    cat = load_catalog() if cat is None else cat
    c = cat.set_index("sequence")
    out = ref.copy()
    hit = out.sequence.isin(c.index)
    out["barcode_id"] = np.where(hit, out.sequence.map(c.barcode_id), "")
    out["ensembl"] = np.where(hit, out.sequence.map(c.ensembl), "")
    out["kind"] = [c.kind.get(s) if s in c.index else kind_of(n) for s, n in zip(out.sequence, out.name)]
    out["catalog_specificity"] = np.where(hit, out.sequence.map(c.specificity), "")
    return out


def detect_offset(r2_seqs: Sequence[bytes], catalog_seqs: Sequence[str], offsets=OFFSETS) -> Dict[int, float]:
    """Fraction of reads whose R2[off:off+L] is exactly a panel / catalog barcode, per offset (L = the most common
    barcode length; barcodes of other lengths are left out of this test)."""
    lens = pd.Series([len(x) for x in catalog_seqs]).value_counts()
    L = int(lens.index[0])
    codes = np.sort(S.encode_strings([x for x in catalog_seqs if len(x) == L], L)[0])
    out = {}
    for off in offsets:
        w = [s[off:off + L] for s in r2_seqs if len(s) >= off + L]
        if not w:
            out[off] = 0.0
            continue
        c, bad = S.encode(S.as_bytes_matrix(w, L))
        out[off] = float(((bad == 0) & (S.lookup(codes, c) >= 0)).sum() / len(r2_seqs))
    return out


SPECIES = {"homo sapiens": "human", "mus musculus": "mouse", "rattus norvegicus": "rat"}
IDENTITY_MAX = 0.10  # reads on catalog barcodes whose reactivity excludes the organism, or that were reused


def discover(feature_seqs: Sequence[bytes], cat: pd.DataFrame, organism: str = "") -> Panel:
    """Identify the panel from feature-barcode windows (already at the chosen offset) of reads in called cells.

    Identity checks (``notes``): ``reactivity_mismatch_share``, the share of panel reads on barcodes whose catalog
    reactivity excludes the sample's organism (a mouse-only antibody in a human sample means the sequence names something
    else in this panel), and ``reused_share``, the share on sequences a published panel used for another antibody
    (``reused`` column). Either above ``IDENTITY_MAX`` makes the status ``ambiguous``: the names need a study reference."""
    L = int(cat.sequence.str.len().max())
    w = [s for s in feature_seqs if len(s) == L]
    n = len(w)
    u, cnt = np.unique(np.array(w, dtype=f"S{L}"), return_counts=True)
    mat = u.view(np.uint8).reshape(len(u), L)
    from .count import FeatureIndex

    fx = FeatureIndex(list(cat.sequence))
    idx, _ = fx.assign(mat)
    support = np.bincount(idx[idx >= 0], weights=cnt[idx >= 0], minlength=len(cat))
    matched = float(support.sum())
    keep = (support >= MIN_READS) & (support >= MIN_SHARE * max(matched, 1))
    feats = cat[keep].copy()
    feats["reads_discovery"] = support[keep].astype(np.int64)
    feats = feats.sort_values("reads_discovery", ascending=False)
    # organism filter is advisory only: a mouse-reactive clone in a human sample is reported, never dropped
    unm = idx == -1
    um_c = cnt[unm]
    um_s = u[unm]
    order = np.argsort(-um_c)
    abundant = [(um_s[i].decode(), int(um_c[i])) for i in order[:200] if um_c[i] >= ABUNDANT_SHARE * n]
    panel_seqs = list(cat.sequence[keep]) if keep.any() else []
    kinds = {q: classify_unmatched(q, panel_seqs) for q, _ in abundant}
    other = [(q, c) for q, c in abundant if kinds[q] == "other"]
    share = {k: sum(c for q, c in abundant if kinds[q] == k) / max(n, 1) for k in ("low_complexity", "shifted", "other")}
    notes = {"reads": n, "matched_share": matched / max(n, 1), "unexplained_share": share["other"],
             "low_complexity_share": share["low_complexity"], "shifted_share": share["shifted"],
             "unexplained_top": other[:20], "n_catalog_hits": int(keep.sum())}
    sp = SPECIES.get(str(organism).split("|")[0].strip().lower(), "")
    sup = feats.reads_discovery.astype(float)
    tot = max(sup.sum(), 1.0)
    react = feats.reactivity.fillna("").str.lower() if "reactivity" in feats else pd.Series("", index=feats.index)
    mism = (react != "") & ~react.str.contains(sp) & (feats.kind != "isotype") if sp else pd.Series(False, index=feats.index)
    reused = feats.reused.fillna("") != "" if "reused" in feats else pd.Series(False, index=feats.index)
    notes["reactivity_mismatch_share"] = float(sup[mism].sum() / tot)
    notes["reactivity_mismatch"] = [f"{n} ({b}, {r})" for n, b, r in zip(feats.specificity[mism], feats.barcode_id[mism],
                                                                        react[mism])][:20]
    notes["reused_share"] = float(sup[reused].sum() / tot)
    notes["reused"] = list(feats.reused[reused])[:20]
    if not keep.any():
        status = "unresolved"
    elif notes["reactivity_mismatch_share"] > IDENTITY_MAX or notes["reused_share"] > IDENTITY_MAX:
        status = "ambiguous"
    elif share["other"] > UNEXPLAINED_MAX:
        status = "incomplete"
    else:
        status = "ok"
    out = pd.DataFrame({
        "id": [b or s for b, s in zip(feats.barcode_id, feats.sequence)],
        "name": feats.specificity.values, "sequence": feats.sequence.values, "kind": feats.kind.values,
        "barcode_id": feats.barcode_id.values, "ensembl": feats.ensembl.values, "clone": feats.clone.values,
        "reactivity": feats.reactivity.values, "reads_discovery": feats.reads_discovery.values,
        "reused": feats.reused.values if "reused" in feats else ""})
    out["source"] = "catalog"
    _dedupe_names(out)
    return Panel(out.reset_index(drop=True), "catalog", -1, status, notes)


def classify_unmatched(q: str, panel_seqs: Sequence[str]) -> str:
    """``low_complexity`` (one base >= 80%, e.g. NovaSeq poly-G of empty clusters, poly-A), ``shifted`` (a panel barcode
    read one base early or late: the 10-nt handle had an indel; Cell Ranger's fixed pattern drops these too) or ``other``
    (an unknown feature barcode: evidence the panel is incomplete)."""
    if max(q.count(b) for b in "ACGTN") >= 0.8 * len(q):
        return "low_complexity"
    for p in panel_seqs:
        L = min(len(p), len(q))
        if q[1:L] == p[:L - 1] or q[:L - 1] == p[1:L]:
            return "shifted"
    return "other"


def _dedupe_names(df: pd.DataFrame):
    """Names must be unique within a panel: two clones of one target get their barcode id appended."""
    dup = df.name.duplicated(keep=False)
    df.loc[dup, "name"] = df.loc[dup, "name"] + " (" + df.loc[dup, "id"] + ")"


def split_kinds(panel: pd.DataFrame):
    """Row indices of antibody+isotype features (``adt_counts``) and of hashtags (``hto_counts``)."""
    h = np.flatnonzero(panel.kind.values == "hashtag")
    a = np.flatnonzero(panel.kind.values != "hashtag")
    return a, h

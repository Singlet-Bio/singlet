# SPDX-License-Identifier: MIT
"""Build the antibody-oligo barcode catalog used to identify panels from the reads.

Sources (public copies of vendor lists; BioLegend's own site is behind a bot challenge and is not scraped):

* ``biolegend_2020``: BioLegend's TotalSeq export (A/B/C, 915 conjugates, with clone, reactivity and Ensembl gene),
  as committed in theislab/scvelo_notebooks ``data/biolegend/totalseq.csv`` (2020-06-25).
* ``rhumba_2018``: rnabioco/rhumba ``CITEseq/dbases/Totalseq_111518.csv`` (TotalSeq-A incl. hashtags, 2018-11-15).
* ``hubmap_{A,B,C}``: hubmapconsortium/citeseq-pipeline ``adt_barcode/adt_{A,B,C}.tsv``.
* ``jax_b_universal``: TheJacksonLaboratory/nf-tenx ``assets/totalseq-b_universal.csv``.
* ``cellgeni_c_universal_v1`` / ``cellgeni_hto``: cellgeni/cellranger_cite_hash feature references.

TotalSeq barcode numbers share one sequence across the A, B and C formats (the format is the capture handle, not the
barcode), so the catalog is keyed by the 15-nt sequence. Every spelling of a sequence's target is kept in ``aliases``;
on 2026-10-23 the sequences with several spellings were all synonyms of one target (CD184 / CXCR4, Biotin /
Streptavidin, Greek vs Latin letters), so no sequence names two different antigens.

    python -P bench/adt/build_catalog.py SRC_DIR python/singlet/adt/data/totalseq_catalog.tsv
"""

import html
import os
import re
import sys
import unicodedata

import pandas as pd


def _clean(x):
    return re.sub(r'^="?|"$', "", str(x)).strip().strip('"') if pd.notna(x) else ""


def load(src):
    rows = []
    d = pd.read_csv(os.path.join(src, "biolegend_theis.csv"), dtype=str)
    for r in d.itertuples(index=False):
        cat, _, bc, spec, clone, react, seq, ens = (_clean(v) for v in r)
        fmt = cat[-1] if cat else ""
        rows.append(dict(sequence=seq, barcode_id=f"{fmt}{bc}" if bc else "", formats=fmt, specificity=spec,
                         clone=clone, reactivity=react if react != "nan" else "", ensembl=ens if ens != "nan" else "",
                         source="biolegend_2020"))
    d = pd.read_csv(os.path.join(src, "rhumba.csv"), dtype=str)
    for r in d.itertuples(index=False):
        m = re.search(r"_([ABC]\d{4})$", r.name)
        rows.append(dict(sequence=r.sequence, barcode_id=m.group(1) if m else "", formats=m.group(1)[0] if m else "A",
                         specificity=r.id.replace("_", " "), clone="", reactivity="", ensembl="", source="rhumba_2018"))
    for fmt in "ABC":
        d = pd.read_csv(os.path.join(src, f"hubmap_{fmt}.tsv"), sep="\t", header=None, names=["name", "sequence"],
                        dtype=str)
        for r in d.itertuples(index=False):
            spec, _, clone = r.name.rpartition("_") if "_" in r.name else (r.name, "", "")
            rows.append(dict(sequence=r.sequence, barcode_id="", formats=fmt, specificity=spec, clone=clone,
                             reactivity="", ensembl="", source=f"hubmap_{fmt}"))
    d = pd.read_csv(os.path.join(src, "jax_b_univ.csv"), dtype=str)
    for r in d.itertuples(index=False):
        rows.append(dict(sequence=r.tag_sequence, barcode_id=r.tag_id, formats="B", specificity=r.tag_name, clone="",
                         reactivity="Human", ensembl="", source="jax_b_universal"))
    d = pd.read_csv(os.path.join(src, "c_univ_v1.csv"), dtype=str)
    for r in d.itertuples(index=False):
        m = re.search(r"([ABC]\d{4})", r.id)
        rows.append(dict(sequence=r.sequence, barcode_id=m.group(1) if m else "", formats="C",
                         specificity=re.sub(r"^Hu\.", "", r.name), clone="", reactivity="Human", ensembl="",
                         source="cellgeni_c_universal_v1"))
    d = pd.read_csv(os.path.join(src, "hto.csv"), dtype=str)
    for r in d.itertuples(index=False):
        rows.append(dict(sequence=r.sequence, barcode_id="", formats="A", specificity=f"Hashtag {r.name[3:]}",
                         clone="", reactivity="Human", ensembl="", source="cellgeni_hto"))
    return pd.DataFrame(rows)


_GREEK = str.maketrans({"α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "κ": "k", "λ": "l"})


def norm_spec(s):
    """Comparable form of a specificity: 'CD279 (PD-1)' / 'CD279' / 'Hu.CD279' -> 'cd279'."""
    s = html.unescape(s).replace("<sup>", "").replace("</sup>", "")
    s = s.translate(_GREEK)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    if s.lower() in ("", "nan"):
        return ""
    if "isotype" in s.lower():
        return "isotype"
    s = re.sub(r"\(.*?\)", "", s.lower())
    s = re.sub(r"^(hu|mo|ms|human|mouse|anti)[ ._-]+", "", s)
    s = s.replace("hashtag", "hashtag ").replace("_", " ")
    return re.sub(r"[^a-z0-9]+", "", s)


def main(src, out):
    d = load(src)
    d = d.fillna("")
    d["sequence"] = d.sequence.str.upper().str.strip()
    d = d[d.sequence.str.fullmatch(r"[ACGT]{15}")]
    d["key"] = d.specificity.map(norm_spec)
    g = []
    for seq, x in d.groupby("sequence", sort=True):
        bl = x[x.source == "biolegend_2020"]
        best = bl.iloc[0] if len(bl) else x.iloc[0]
        keys = sorted(set(x.key) - {""})
        hashtag = any("hashtag" in k for k in keys)
        isotype = "isotype" in keys
        g.append(dict(sequence=seq,
                      barcode_id=next((b for b in x.barcode_id if b), ""),
                      formats="".join(sorted(set("".join(x.formats)))),
                      specificity=html.unescape(best.specificity), clone=best.clone,
                      reactivity=next((v for v in bl.reactivity if v), best.reactivity),
                      ensembl=next((v for v in x.ensembl if v), ""),
                      kind="hashtag" if hashtag else ("isotype" if isotype else "antibody"),
                      aliases="|".join(sorted(set(html.unescape(v) for v in x.specificity) - {"", "nan"})),
                      sources="|".join(sorted(set(x.source)))))
    cat = pd.DataFrame(g)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cat.to_csv(out, sep="\t", index=False)
    print(f"{len(cat)} sequences; kinds {cat.kind.value_counts().to_dict()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

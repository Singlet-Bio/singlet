# SPDX-License-Identifier: MIT
"""Guide libraries and where the protospacer sits in R2.

A guide row is ``(id, name, sequence, target_gene, target_gene_id, kind)``, ``kind`` in ``targeting`` / ``non-targeting``
/ ``safe-targeting``. Sources: a Cell Ranger feature reference (``feature_type == "CRISPR Guide Capture"``), or the
authors' guide table.

**Read geometry.** Guide-capture R2 holds the protospacer next to a constant sequence: the sgRNA scaffold for direct
capture (capture sequence 1/2 in the scaffold stem, 10x 3' v3 "CRISPR"; CROP-seq-like 5' capture) or a vector
sequence for guide-barcode amplicons (Perturb-seq GBC). Cell Ranger encodes this as a pattern, ``(BC)GTTTAAGAGC...``
(protospacer before the constant) or ``^NNNNNNNNNN(BC)`` / ``5PNNN(BC)`` (fixed offset). :func:`parse_pattern` reads both;
:func:`detect_geometry` measures it from the reads when no pattern is known: the scaffold anchor that most reads hold and
the offset of the protospacer (the constant in front of the most frequent windows).
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# sgRNA scaffold starts (sense, right after the protospacer): standard S. pyogenes, the "F+E" optimised scaffold
# (Chen 2013), the CROP-seq / Replogle cs1 scaffold variants. Only the first 12 nt are used as anchors.
SCAFFOLDS = {
    "standard": "GTTTTAGAGCTAGAAATAGCAAGTT",
    "optimized": "GTTTAAGAGCTATGCTGGAAACAGCATAGCAAGTT",
    "cr_default": "GTTTAAGAGCTAAGCTGGAAACAGCATAGCAAGTT",
    "replogle_cs1": "GTTTCAGAGCTATGCTGGAAACAGCATAGCAAGTT",
}
# dual-guide vectors (Replogle 2022): protospacer A before a GTTTCAGAGC scaffold, B before GTTTAAGAGC
DUAL_ANCHORS = ("GTTTCAGAGC|GTTTAAGAGC",)
ANCHOR_LEN = 12


@dataclass(frozen=True)
class Geometry:
    """Where the protospacer is in R2: ``anchor`` (constant sequence right after it, sense) or a fixed ``offset``."""

    length: int
    anchor: Optional[str] = (
        None  # protospacer = R2[pos - length : pos] where R2[pos:] starts with anchor; several
    )
    # anchors separated by "|" (dual-guide vectors: each protospacer has its own scaffold variant)
    offset: Optional[int] = None  # protospacer = R2[offset : offset + length]
    reverse: bool = False  # R2 is antisense (reverse-complement it first)

    def describe(self) -> str:
        where = f"before {self.anchor}" if self.anchor else f"at offset {self.offset}"
        return f"{self.length} nt {where}{' (R2 reverse-complemented)' if self.reverse else ''}"


_RC = bytes.maketrans(b"ACGTN", b"TGCAN")


def rc(s: bytes) -> bytes:
    return s.translate(_RC)[::-1]


def parse_pattern(pattern: str, length: int) -> Geometry:
    """Cell Ranger pattern -> Geometry. ``(BC)CONST`` -> anchor; ``^N*(BC)`` / ``5PN*(BC)`` -> offset."""
    p = pattern.strip()
    m = re.fullmatch(r"\(BC\)([ACGTN|]+)", p)
    if m:
        return Geometry(length, anchor="|".join(x[:ANCHOR_LEN] for x in m.group(1).split("|")))
    m = re.fullmatch(r"(?:\^|5P)?(N*)\(BC\)([ACGTN]*)", p)
    if m:
        return Geometry(length, offset=len(m.group(1)))
    m = re.fullmatch(r"(?:\^|5P)?([ACGTN]+)\(BC\)", p)
    if m:
        return Geometry(length, offset=len(m.group(1)))
    raise ValueError(f"unsupported feature pattern {pattern!r}")


def extract(r2: bytes, g: Geometry) -> Optional[bytes]:
    s = rc(r2) if g.reverse else r2
    if g.anchor is not None:
        for a in g.anchor.split("|"):
            i = s.find(a.encode())
            if i >= g.length:
                return s[i - g.length : i]
        return None
    o = g.offset or 0
    w = s[o : o + g.length]
    return w if len(w) == g.length else None


def detect_geometry(
    r2_sample: Sequence[bytes], length: int = 20, guides: Optional[Sequence[str]] = None
) -> Dict:
    """Measure the geometry from reads: share of reads with each scaffold anchor (both strands), and, with a guide list,
    the share of extracted windows that are guides (exact) for every candidate geometry. Returns the best and the table."""
    n = max(1, len(r2_sample))
    gset = set(x.encode() for x in guides) if guides else None
    cands: List[Tuple[float, float, Geometry]] = []
    for rev in (False, True):
        seqs = [rc(s) for s in r2_sample] if rev else list(r2_sample)
        for sc in [v[:ANCHOR_LEN] for v in SCAFFOLDS.values()] + list(DUAL_ANCHORS):
            g = Geometry(length, anchor=sc, reverse=rev)
            ws = [extract(s, g) for s in seqs]
            found = sum(w is not None for w in ws) / n
            hit = sum(w in gset for w in ws if w is not None) / n if gset else found
            cands.append((hit, found, g))
        if gset:
            for off in range(0, 60):
                g = Geometry(length, offset=off, reverse=rev)
                ws = [s[off : off + length] for s in seqs]
                hit = sum(w in gset for w in ws) / n
                cands.append((hit, hit, g))
    cands.sort(key=lambda x: (-x[0], -x[1]))
    best = cands[0]
    table = [
        {"geometry": c[2].describe(), "guide_rate": round(c[0], 4), "window_rate": round(c[1], 4)}
        for c in cands[:8]
    ]
    return {"geometry": best[2], "guide_rate": best[0], "window_rate": best[1], "candidates": table}


@dataclass
class GuideLibrary:
    ids: List[str]
    names: List[str]
    sequences: List[str]
    target_gene: List[str]
    target_gene_id: List[str]
    kind: List[str]
    pattern: Optional[str] = None
    source: str = ""

    def __len__(self):
        return len(self.ids)

    def check(self):
        if len(set(self.sequences)) != len(self.sequences):
            dup = [s for s, c in Counter(self.sequences).items() if c > 1][:5]
            raise ValueError(f"duplicate guide sequences: {dup}")
        return self

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "id": self.ids,
                "name": self.names,
                "sequence": self.sequences,
                "target_gene": self.target_gene,
                "target_gene_id": self.target_gene_id,
                "kind": self.kind,
            }
        )


_NT_WORDS = (
    "non-targeting",
    "nontargeting",
    "non_targeting",
    "negative_control",
    "neg_ctrl",
    "nt_",
    "scramble",
    "ntc",
)


def guide_kind(name: str, target: str) -> str:
    s = f"{name} {target}".lower()
    if "safe" in s and "target" in s:
        return "safe-targeting"
    if any(w in s for w in _NT_WORDS) or target.strip().lower() in (
        "",
        "non-targeting",
        "nontargeting",
        "non_targeting",
        "nt",
        "ntc",
        "nan",
        "none",
        "control",
        "negative",
        "na",
    ):
        return "non-targeting"
    return "targeting"


def from_feature_reference(path: str, source: str = "feature_reference") -> GuideLibrary:
    """Cell Ranger feature reference CSV, CRISPR rows only."""
    ids, names, seqs, tg, tgid, kinds = [], [], [], [], [], []
    pattern = None
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            ft = (r.get("feature_type") or "").strip()
            if ft and "crispr" not in ft.lower():
                continue
            gid = (r.get("id") or r.get("name")).strip()
            nm = (r.get("name") or gid).strip()
            t = (r.get("target_gene_name") or "").strip()
            ids.append(gid)
            names.append(nm)
            seqs.append(r["sequence"].strip().upper())
            tg.append(t)
            tgid.append((r.get("target_gene_id") or "").strip())
            kinds.append(guide_kind(nm, t))
            pattern = pattern or (r.get("pattern") or "").strip() or None
    return GuideLibrary(ids, names, seqs, tg, tgid, kinds, pattern, source).check()


def from_table(rows: Iterable[Dict[str, str]], source: str) -> GuideLibrary:
    """Rows with ``id``, ``sequence``, ``target_gene`` (and optional ``name``, ``target_gene_id``, ``kind``)."""
    ids, names, seqs, tg, tgid, kinds = [], [], [], [], [], []
    for r in rows:
        ids.append(r["id"])
        names.append(r.get("name") or r["id"])
        seqs.append(r["sequence"].upper())
        tg.append(r.get("target_gene", ""))
        tgid.append(r.get("target_gene_id", ""))
        kinds.append(r.get("kind") or guide_kind(names[-1], tg[-1]))
    return GuideLibrary(ids, names, seqs, tg, tgid, kinds, None, source).check()

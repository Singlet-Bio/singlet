# SPDX-License-Identifier: MIT
"""Guide UMIs per (cell barcode, guide), with the ADT counting core (:func:`singlet.adt.count.count_library`).

The protospacer window is cut out of R2 by the library geometry (:mod:`singlet.crispr.guides`): before the scaffold
anchor (direct capture) or at a fixed offset. Everything after that is the antibody counting logic, which is Cell
Ranger's feature-barcoding logic: exact or unique Hamming-1 guide match, whitelist barcode correction with the
1-mismatch posterior, Cell Ranger ``correct_umis``, molecules on the GEX sample's columns form the matrix and the rest the
ambient profile (empty droplets).

Guides of several lengths (19 and 20 nt libraries) are matched right-aligned to the anchor: windows and guides are
reversed before the index, which matches the first ``L`` bases.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

from singlet.adt import count as AC

from .guides import Geometry, GuideLibrary, extract


def guide_windows(
    pairs: Iterable[Tuple[bytes, bytes]],
    g: Geometry,
    right_aligned: bool,
    tally: Optional[dict] = None,
):
    """(R1, protospacer window) pairs; reads without the anchor give an all-N window (never matched), counted in
    ``tally["reads_without_protospacer"]``."""
    missing = b"N" * g.length
    n = 0
    for s1, s2 in pairs:
        w = extract(s2, g)
        if w is None:
            w = missing
            n += 1
        yield s1, (w[::-1] if right_aligned else w)
    if tally is not None:
        tally["reads_without_protospacer"] = tally.get("reads_without_protospacer", 0) + n


def count_library(
    r1_paths,
    r2_paths,
    *,
    library: GuideLibrary,
    geometry: Geometry,
    cb_len: int,
    umi_len: int,
    whitelist_codes,
    gex_barcodes: Sequence[str],
    translation=None,
    max_reads: Optional[int] = None,
    pairs_factory=None,
    shards: int = 0,
    tmpdir: Optional[str] = None,
) -> AC.LibraryCounts:
    right = geometry.anchor is not None
    L = max(len(s) for s in library.sequences)
    if geometry.length != L:
        geometry = Geometry(L, geometry.anchor, geometry.offset, geometry.reverse)
    seqs = [s[::-1] for s in library.sequences] if right else list(library.sequences)
    base = pairs_factory or (lambda: AC.iter_pairs(r1_paths, r2_paths))
    tally: dict = {}
    lc = AC.count_library(
        r1_paths,
        r2_paths,
        cb_len=cb_len,
        umi_len=umi_len,
        offset=0,
        panel_sequences=seqs,
        whitelist_codes=whitelist_codes,
        gex_barcodes=gex_barcodes,
        translation=translation,
        max_reads=max_reads,
        pairs_factory=lambda: guide_windows(base(), geometry, right, tally),
        shards=shards,
        tmpdir=tmpdir,
    )
    lc.features = list(library.ids)
    if right:
        lc.unmatched = [(s[::-1], c) for s, c in lc.unmatched]
    lc.unmatched = [(s, c) for s, c in lc.unmatched if set(s) != {"N"}]
    lc.stats["reads_without_protospacer"] = int(tally.get("reads_without_protospacer", 0))
    lc.stats["geometry"] = geometry.describe()
    return lc

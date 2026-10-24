# SPDX-License-Identifier: MIT
"""Read geometry of antibody-capture libraries.

Feature-barcode reads keep the cell barcode and UMI of the gene-expression library in R1; R2 holds the antibody
barcode: at position 0 for TotalSeq-A (poly-A capture, pattern ``^(BC)``) and after 10 nt for TotalSeq-B (3' capture
sequence 1) and TotalSeq-C (5' template switch), pattern ``5PNNNNNNNNNN(BC)``. The offset is measured from the reads
(:func:`singlet.adt.panel.detect_offset`), not assumed.

Whitelists are the 10x lists the GEX pipeline uses. On 3' v3/v3.1 (and v4) gel beads, the oligos that capture
TotalSeq-B (capture sequence 1/2) carry a different barcode than the poly-dT oligo of the same bead; Cell Ranger maps
them with ``translation/3M-february-2018.txt.gz`` (``translation/3M-3pgex-may-2023.txt.gz`` for v4). TotalSeq-A on v3
is captured by poly-dT, so it needs no translation. Whether a library needs it is decided from the reads too
(:func:`singlet.adt.count.choose_translation`): the barcode overlap with the GEX cells with and without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# GitHub copies of the 10x lists (the compute nodes reach GitHub; 10x's own site needs a login for some files).
WL_BASE = "https://raw.githubusercontent.com/Teichlab/scg_lib_structs/master/data/10X-Genomics/"


@dataclass(frozen=True)
class Chemistry:
    name: str
    cb_len: int
    umi_len: int
    whitelist: Optional[str]  # file name under WL_BASE, None = no whitelist (barcodes taken from the GEX columns)
    translation: Optional[str] = None  # feature-barcode -> GEX barcode table, used only when the reads ask for it
    supported: bool = True


CHEMISTRIES = {
    "10xv2": Chemistry("10xv2", 16, 10, "737K-august-2016.txt.gz"),
    "10xv3": Chemistry("10xv3", 16, 12, "3M-february-2018.txt.gz", "translation_3M-february-2018.txt.gz"),
    "10xv4": Chemistry("10xv4", 16, 12, "3M-3pgex-may-2023.txt.gz", "translation_3M-3pgex-may-2023.txt.gz"),
    "10x5v1": Chemistry("10x5v1", 16, 10, "737K-august-2016.txt.gz"),
    "10x5v2": Chemistry("10x5v2", 16, 10, "737K-august-2016.txt.gz"),
    "10x5v3": Chemistry("10x5v3", 16, 12, "3M-5pgex-jan-2023.txt.gz"),
    "10x-arc-gex": Chemistry("10x-arc-gex", 16, 12, None, supported=False),
    "bd-rhapsody": Chemistry("bd-rhapsody", 0, 0, None, supported=False),
    "dropseq": Chemistry("dropseq", 12, 8, None, supported=False),
}

# atlas protocol names -> chemistry. The atlas calls 5' v1/v2 libraries "10xv2" (same barcodes, same 16+10 R1), which is
# right for counting: only the R2 offset differs, and that is measured.
PROTOCOL_ALIASES = {
    "10xv2": "10xv2", "10xv3-5p": "10x5v3", "10xv3": "10xv3", "10xv3.1": "10xv3", "10xv4": "10xv4", "10x5": "10x5v2", "10x5v1": "10x5v1",
    "10x5v2": "10x5v2", "10x-5p": "10x5v2", "10x5p": "10x5v2",
}


def for_protocol(protocol: str) -> Chemistry:
    p = (protocol or "").strip().lower()
    if p in PROTOCOL_ALIASES:
        return CHEMISTRIES[PROTOCOL_ALIASES[p]]
    if p in CHEMISTRIES:
        return CHEMISTRIES[p]
    return Chemistry(p or "unknown", 0, 0, None, supported=False)


def from_read_length(r1_len: int) -> Optional[str]:
    """10x protocol family from the R1 length when the atlas protocol is missing: 26 nt = 16+10 (v2 / 5'), 28 = 16+12 (v3)."""
    if r1_len in (26, 27):
        return "10xv2"
    if 28 <= r1_len <= 31:
        return "10xv3"
    return None

# SPDX-License-Identifier: MIT
"""Research Alpha QC: one row per sample of every manifest the sweep cached (labels, provenance, called_ed, sizes).

    python qc_manifest_rows.py MANIFEST_DIR OUT.tsv.gz
"""

import glob
import gzip
import json
import os
import sys


def main():
    d, out = sys.argv[1], sys.argv[2]
    cols = ["gse", "gsm", "build", "n_columns", "n_called", "bytes", "has_annotations", "ann_n_rows", "label_columns",
            "reads_used", "reads_used_source", "wrong_strand_fraction", "strand_class", "library_type", "called_ed",
            "called_ed_validated", "n_called_ed", "called_source", "junction_support", "legacy_bytes", "n_members",
            "cells_columns"]
    with gzip.open(out, "wt") as fh:
        fh.write("\t".join(cols) + "\n")
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            m = json.load(open(f))
            for s in m.get("samples") or []:
                pv = s.get("provenance") or {}
                stv = pv.get("strand") or {}
                cm = (s.get("members") or {}).get("cells") or {}
                ced = cm.get("called_ed") or {}
                ann = s.get("annotations") or {}
                a0 = next(iter(ann.values()), {}) if ann else {}
                lab = cm.get("labels") or {}
                row = [m.get("gse"), s.get("gsm"), m.get("reference_build"), s.get("n_columns"), s.get("n_called"),
                       s.get("bytes"), int(bool(ann)), a0.get("n_rows"), ",".join(lab.get("columns") or []),
                       pv.get("reads_used"), pv.get("reads_used_source"), stv.get("wrong_strand_fraction"), stv.get("class"),
                       (pv.get("library_type") or {}).get("verdict"), int("called_ed" in (cm.get("columns") or [])),
                       ced.get("validated"), ced.get("n_called_ed"), cm.get("called_source"),
                       int(bool(s.get("junction_support"))), (m.get("legacy") or {}).get("bytes"),
                       len(s.get("members") or {}), ",".join(cm.get("columns") or [])]
                fh.write("\t".join("" if x is None else str(x).replace("\t", " ") for x in row) + "\n")


if __name__ == "__main__":
    main()

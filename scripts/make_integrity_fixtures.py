#!/usr/bin/env python3
"""Write the small .singlet files the integrity tests (R and Python) use.

    python scripts/make_integrity_fixtures.py

Output goes to r/tests/testthat/fixtures/: one good bundle and one each that is
truncated, has a member that fails its checksum, or declares a newer schema.
"""
import hashlib
import io
import json
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "r" / "tests" / "testthat" / "fixtures"
MEMBERS = {
    "feature_vocab.json": b'{"genes": []}',
    "samples/GSM1/exon_counts.1pz": bytes(range(256)) * 8,
    "samples/GSM1/summary.json": b'{"n_cells": 1}',
}


def bundle(schema="1.0", bad_checksum=False) -> bytes:
    sums = {n: hashlib.sha256(d).hexdigest() for n, d in MEMBERS.items()}
    if bad_checksum:
        sums["samples/GSM1/summary.json"] = "0" * 64
    manifest = {"schema_version": schema, "gse_id": "GSE1", "gsm_ids": ["GSM1"], "checksums": sums}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", allowZip64=True) as zf:
        for n, d in MEMBERS.items():
            zf.writestr(zipfile.ZipInfo(n, (2026, 1, 1, 0, 0, 0)), d, compress_type=zipfile.ZIP_STORED)
        zf.writestr(zipfile.ZipInfo("manifest.json", (2026, 1, 1, 0, 0, 0)), json.dumps(manifest))
    return buf.getvalue()


OUT.mkdir(parents=True, exist_ok=True)
good = bundle()
(OUT / "good.singlet").write_bytes(good)
(OUT / "truncated.singlet").write_bytes(good[: len(good) // 2])
(OUT / "bad_checksum.singlet").write_bytes(bundle(bad_checksum=True))
(OUT / "newer_schema.singlet").write_bytes(bundle(schema="2.0"))
print("wrote", sorted(p.name for p in OUT.iterdir()))

# SPDX-License-Identifier: MIT
"""Build the shipped packs (python/singlet/v2/packs/) from the measured study inputs.

    python -m singlet.v2.packs_build --vocab-src bench/compression/corpus_w4_vocab --whitelists python/singlet/data/whitelists \\
        --out python/singlet/v2/packs [--svp bench/compression/results/w4_vocab_packs]

Inputs (not shipped): for every reference build the row-name metadata blocks of one representative sample plus the study's
``feature_vocab.json`` (fetched by bench/compression/tools/w4_vocab_fetch.py) and the 10x whitelist text files.

Every pack is verified before it is written: the expansion equals the source names byte for byte, and, when the old ``.svp`` bodies are
given, the new SVP0 body is compared with them. All fetched studies of a build are then compared with the expansion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import struct
import time

from . import packs as P
from . import util

WHITELISTS = [  # (file stem, aliases) - the 10x lists that real samples use
    "3M-february-2018",
    "737K-august-2016",
    "737K-april-2014",
    "gex_737K-arc-v1",
]
ALIAS = {"737K-august-2016": ["737K-august-2016", "737K-cratac-v1"]}
SPECIES_TAG = {"GRCh38-2024-A": "GRCh38___", "GRCm39-2024-A": "GRCm39___"}
BARNYARD = "GRCh38-GRCm39-barnyard"


def _parse_tlv(raw: bytes):
    rows = []
    p = 0
    while p < len(raw):
        tag = raw[p]
        p += 1
        if tag == 0:
            break
        (ln,) = struct.unpack_from("<I", raw, p)
        p += 4
        body = raw[p : p + ln]
        p += ln
        if tag == 1:
            parts = body.split(b"\0")
            assert parts[-1] == b""
            rows.extend(x.decode("utf-8") for x in parts[:-1])
    return rows


def load_meta_rows(path: str):
    return _parse_tlv(util.zstd_decompress(open(path, "rb").read()))


def build_all(vocab_src: str, wl_dir: str, out: str, svp_dir: str | None = None, log=print) -> list:
    os.makedirs(out, exist_ok=True)
    index = list(csv.DictReader(open(os.path.join(vocab_src, "index.tsv")), delimiter="\t"))
    by_build: dict[str, list] = {}
    for r in index:
        by_build.setdefault(r["build"], []).append(r)
    entries = []
    packs_bytes: dict[str, dict] = {}  # build -> {"exon": bytes, "intron": bytes}

    def emit(buf: bytes, kind_name: str, **extra):
        e = P.index_entry(buf, kind_name, extra)
        with open(os.path.join(out, e["id"] + ".pack"), "wb") as fh:
            fh.write(buf)
        entries.append(e)
        log(
            "  %-9s %-24s %-7s %9d B  id %s"
            % (
                kind_name,
                extra.get("build", extra.get("protocol", "")),
                extra.get("member", ""),
                len(buf),
                e["id"],
            )
        )
        return e

    singles = [b for b in by_build if "barnyard" not in b]
    for build in singles:
        r0 = by_build[build][0]
        gse, gsm = r0["gse"], r0["gsm"]
        ex = load_meta_rows(os.path.join(vocab_src, "%s.%s.exon.meta" % (gse, gsm)))
        it = load_meta_rows(os.path.join(vocab_src, "%s.%s.intron.meta" % (gse, gsm)))
        vraw = open(os.path.join(vocab_src, "%s.feature_vocab.json" % gse), "rb").read()
        vocab = json.loads(vraw)
        t0 = time.time()
        S = P.build_struct(ex, it, vocab)
        body = P.encode_struct(S)
        if svp_dir and os.path.exists(os.path.join(svp_dir, build + ".svp")):
            old = open(os.path.join(svp_dir, build + ".svp"), "rb").read()
            log(
                "  %s: SVP0 body %s the measured w4_vocab pack (%d B)"
                % (build, "IDENTICAL to" if old == body else "DIFFERS from", len(old))
            )
        exon_pack = P.build_exon_pack(S, {"source": "%s/%s" % (gse, gsm)})
        intron_pack = P.build_intron_pack(exon_pack)
        # verification: pack -> names byte exact, vocab json byte exact
        store = P.PackStore(dirs=[out], download=False)
        e1 = emit(
            exon_pack,
            "vocab",
            build=build,
            member="exon",
            n=len(ex),
            ship=build in ("GRCh38-2024-A", "GRCm39-2024-A"),
        )
        e2 = emit(
            intron_pack,
            "vocab",
            build=build,
            member="intron",
            n=len(it),
            ship=build in ("GRCh38-2024-A", "GRCm39-2024-A"),
        )
        store.rescan()
        pe, pi = store.get(e1["id"]), store.get(e2["id"])
        assert pe.names() == ex, "exon expansion differs"
        assert pi.names() == it, "intron expansion differs"
        assert pe.vocab_json() == vraw, "feature_vocab.json differs"
        packs_bytes[build] = {"exon": exon_pack, "intron": intron_pack}
        # lazy lookups agree with the full expansion
        import numpy as np

        rng = np.random.default_rng(1)
        ii = rng.integers(0, len(ex), 2000)
        assert pe.names_at(ii) == [ex[i] for i in ii]
        jj = rng.integers(0, len(it), 2000)
        assert pi.names_at(jj) == [it[i] for i in jj]
        log("  %s: exon/intron/vocab_json verified (%.1fs)" % (build, time.time() - t0))
    # barnyard composite
    if BARNYARD in by_build:
        r0 = by_build[BARNYARD][0]
        gse, gsm = r0["gse"], r0["gsm"]
        ex = load_meta_rows(os.path.join(vocab_src, "%s.%s.exon.meta" % (gse, gsm)))
        it = load_meta_rows(os.path.join(vocab_src, "%s.%s.intron.meta" % (gse, gsm)))
        vraw = open(os.path.join(vocab_src, "%s.feature_vocab.json" % gse), "rb").read()
        h, m = "GRCh38-2024-A", "GRCm39-2024-A"
        parts = {}
        for member, names in (("exon", ex), ("intron", it)):
            comp = [(P.pack_id(packs_bytes[b][member]), SPECIES_TAG[b]) for b in (h, m)]
            parts[member] = P.build_composite(
                BARNYARD, member, comp, len(names), hashlib.sha256(util.join_nul(names)).hexdigest()
            )
        e1 = emit(parts["exon"], "vocab", build=BARNYARD, member="exon", n=len(ex), ship=False)
        e2 = emit(parts["intron"], "vocab", build=BARNYARD, member="intron", n=len(it), ship=False)
        store = P.PackStore(dirs=[out], download=False)
        pe, pi = store.get(e1["id"]), store.get(e2["id"])
        assert pe.names() == ex, "barnyard exon expansion differs"
        assert pi.names() == it, "barnyard intron expansion differs"
        assert pe.vocab_json() == vraw, "barnyard feature_vocab.json differs"
        log("  %s: composite verified" % BARNYARD)
    # cross-study check
    store = P.PackStore(dirs=[out], download=False)
    bad = 0
    n = 0
    for r in index:
        path = os.path.join(vocab_src, "%s.%s.%s.meta" % (r["gse"], r["gsm"], r["kind"]))
        if not os.path.exists(path):
            continue
        rows = load_meta_rows(path)
        pk = store.find_vocab(r["kind"], rows, r["build"])
        ok = pk is not None and pk.build == r["build"]
        n += 1
        bad += 0 if ok else 1
        if not ok:
            log("  MISMATCH %s %s %s" % (r["gse"], r["gsm"], r["kind"]))
    log("  cross-study: %d/%d (study, matrix) name lists equal a pack" % (n - bad, n))
    # whitelists
    for stem in WHITELISTS:
        txt = [ln.strip() for ln in open(os.path.join(wl_dir, stem + ".txt")) if ln.strip()]
        names = ALIAS.get(stem, [stem])
        buf = P.build_whitelist(txt, {"protocol": stem, "names": names})
        w = P.Whitelist(buf)
        assert w.all_barcodes() == sorted(set(txt)), "whitelist roundtrip failed"
        emit(buf, "whitelist", protocol=stem, aliases=names, n=w.n, k=w.k, ship=True)
    entries.sort(
        key=lambda e: (e["kind"], e.get("build", ""), e.get("member", ""), e.get("protocol", ""))
    )
    P.write_index(entries, os.path.join(out, "index.json"))
    log(
        "wrote %d packs, %d bytes total, index.json"
        % (len(entries), sum(e["bytes"] for e in entries))
    )
    return entries


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--vocab-src", default="bench/compression/corpus_w4_vocab")
    ap.add_argument("--whitelists", default="python/singlet/data/whitelists")
    ap.add_argument(
        "--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "packs")
    )
    ap.add_argument("--svp", default="bench/compression/results/w4_vocab_packs")
    a = ap.parse_args(argv)
    build_all(a.vocab_src, a.whitelists, a.out, a.svp)


if __name__ == "__main__":
    main()

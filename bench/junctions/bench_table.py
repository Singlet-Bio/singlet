# SPDX-License-Identifier: MIT
"""Junction table benchmark on a local mirror of a migrated study (read-only): table sizes per tier, decode and lookup speed, the
per-sample cost of junction rows as table ids (+ local junctions + legacy permutation) against inline names, and exactness.

    python -P bench/junctions/bench_table.py <mirror root> [GSE] > bench/junctions/results_<GSE>.json

The mirror is what ``singlet.store.client.download_study`` writes. Tiers for one study (the atlas build uses studies, not samples):
core = both ends on annotated exon boundaries (exon-pack proxy; the GTF decides at build time) or seen in >= 3 samples;
rare = seen in exactly 2 samples; local = seen in one sample only (stays in that sample object).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "python")
from singlet import _pz2  # noqa: E402
from singlet.store import client  # noqa: E402
from singlet.store import junctions as J  # noqa: E402
from singlet.store import packs as P  # noqa: E402
from singlet.store.packs import default_store  # noqa: E402

root = sys.argv[1]
gse = sys.argv[2] if len(sys.argv) > 2 else "GSE145370"
BUILD = "GRCh38-2024-A"
st = client.open_study(gse, base=root)
out: dict = {"study": gse, "samples": len(st.gsm_ids)}


def ms(f, reps=15):
    """Median CPU milliseconds (process time: less sensitive than wall time to other load on the machine) and wall milliseconds."""
    c, w = [], []
    for _ in range(reps):
        a, b = time.process_time(), time.perf_counter()
        f()
        c.append(time.process_time() - a)
        w.append(time.perf_counter() - b)
    return {
        "cpu_ms": round(1000 * float(np.median(c)), 2),
        "wall_ms": round(1000 * float(np.median(w)), 2),
    }


# ---- the samples' junction lists and matrices
samples = []
for g in st.gsm_ids:
    sm = st.samples[g]
    m = sm.members.get("sj_counts.1pz")
    if m is None or m.kind != 1:
        continue
    src = st.source_for(sm)
    blob = bytes(src.read(m.off, m.len))
    src.close()
    mem = _pz2.Member(blob)
    samples.append((g, blob, mem, mem.row_names()["names"]))

seen = Counter()
for _g, _b, _m, names in samples:
    seen.update(set(names))
allj = J.parse_names(list(seen))
ann = J.annotation_from_exon_pack(default_store().vocab_packs("exon", BUILD)[0])
allj = J.annotate(allj, ann)
nsamp = np.array([seen[x] for x in allj.names()])
proxy_ann = (allj.attr & (J.DONOR_ANN | J.ACCEPTOR_ANN)) == (J.DONOR_ANN | J.ACCEPTOR_ANN)
core = proxy_ann | (nsamp >= 3)
rare = ~core & (nsamp == 2)
local = ~core & ~rare
out["junctions"] = {
    "union": len(allj),
    "core": int(core.sum()),
    "rare": int(rare.sum()),
    "local_only": int(local.sum()),
    "core_proxy_annotated": int(proxy_ann.sum()),
}

# ---- table packs
seg_core = J.build_segment(allj.take(np.flatnonzero(core)), build=BUILD, id_lo=0, tier="core")
seg_rare = J.build_segment(
    allj.take(np.flatnonzero(rare)), build=BUILD, id_lo=int(core.sum()), tier="rare"
)
ver = J.build_version([seg_core, seg_rare], build=BUILD, version=1)
out["table_bytes"] = {
    "core": len(seg_core),
    "rare": len(seg_rare),
    "version": len(ver),
    "core_bytes_per_junction": round(len(seg_core) / max(int(core.sum()), 1), 3),
    "names_zstd19_bytes_for_comparison": len(
        __import__("zstandard")
        .ZstdCompressor(level=19)
        .compress("\n".join(allj.take(np.flatnonzero(core)).names()).encode())
    ),
}

# per-stream sizes of the core segment, and the SCX1-style alternative (FOR bit-packing in 65,536-row groups, no zstd)
_k, _h, body = P.parse(seg_core)
p = 4
streams = {}
for nm in ("chrom_runs", "start_delta", "length", "attr", "gene", "n_tx"):
    a, p2 = J._dec_stream(body, p)
    streams[nm] = (p2 - p, a)
    p = p2
out["core_streams_bytes"] = {k: v[0] for k, v in streams.items()}


def for_bytes(v, group=65536):
    v = np.asarray(v, np.int64)
    tot = 0
    for i in range(0, len(v), group):
        g = v[i : i + group]
        bits = int(np.max(g) - np.min(g)).bit_length() if len(g) else 0
        tot += 24 + (len(g) * bits + 7) // 8
    return tot


out["core_scx1_for_bytes"] = {
    k: for_bytes(a) for k, (_n, a) in streams.items() if k != "chrom_runs"
}
out["core_scx1_for_total"] = sum(out["core_scx1_for_bytes"].values())

# ---- speed: full decode of the core, id -> coordinates, coordinates -> id
out["decode_core_ms"] = ms(lambda: J.decode_body(body, list(_h["chroms"])))
tab = J.JunctionTable(ver, [seg_core, seg_rare])
rng = np.random.default_rng(0)
q = rng.integers(0, tab.n, 1_000_000)
tab.lookup(q[:10])
out["lookup_1M_random_ids_ms"] = ms(lambda: tab.lookup(q), reps=5)
out["names_100k_ids_ms"] = ms(lambda: tab.names(q[:100_000]), reps=3)
probe = tab.lookup(q[:100_000])
tab.find(probe.take(np.arange(10)))
out["find_100k_coords_ms"] = ms(lambda: tab.find(probe), reps=3)
assert np.array_equal(tab.find(probe), q[:100_000]), "find is not the inverse of lookup"

# a larger core for the speed target: the pilot core replicated onto 6 sets of renamed chromosomes (~6x rows)
big = []
for r in range(6):
    jc = allj.take(np.flatnonzero(core))
    big.append(
        J.Junctions(
            [f"{c}_r{r}" for c in jc.chroms], jc.chrom, jc.start, jc.end, jc.attr, jc.gene, jc.n_tx
        )
    )
bj = J.Junctions(
    sum((b.chroms for b in big), []),
    np.concatenate([b.chrom.astype(np.int64) + i * len(big[0].chroms) for i, b in enumerate(big)]),
    np.concatenate([b.start for b in big]),
    np.concatenate([b.end for b in big]),
    np.concatenate([b.attr for b in big]),
    np.concatenate([b.gene for b in big]),
    np.concatenate([b.n_tx for b in big]),
)
bseg = J.build_segment(bj, build=BUILD, id_lo=0, tier="core")
_bk, bh, bbody = P.parse(bseg)
out["scaled_core"] = {
    "junctions": len(bj),
    "bytes": len(bseg),
    "decode_ms": ms(lambda: J.decode_body(bbody, list(bh["chroms"])), reps=7),
}

# ---- per sample: rows as ids + local junctions + legacy permutation, against inline names; exact regeneration
per = []
tot = Counter()
for g, blob, mem, names in samples:
    h = mem.header
    rows = J.sample_rows(tab, names)
    r = mem.decode(mode="canonical", threads=0, dtype="uint32")
    csc = sp.csc_matrix((r["data"], r["indices"], r["indptr"]), shape=(h["m"], h["n"]))
    inv = np.empty(len(rows.legacy_perm), np.int64)
    inv[rows.legacy_perm] = np.arange(len(rows.legacy_perm))
    v2 = csc[rows.legacy_perm, :].tocsc()
    v2.sort_indices()
    newblob, _ = _pz2.encode(
        v2.indptr.astype(np.int64),
        v2.indices.astype(np.uint32),
        v2.data.astype(np.uint32),
        h["m"],
        h["n"],
        codec="auto",
        chunk_cols=h["chunk_cols"],
        compute_digest=False,
        row_names={"kind": "pack_subset", "pack_id": tab.id[:8], "pack_index": rows.ids},
        col_names={"kind": "cells", "names": [str(i) for i in range(h["n"])]},
    )
    loc = J.encode_local(rows.local)
    perm = J.encode_perm(rows.legacy_perm)
    # exactness: names and the legacy matrix come back from the new member + table + local list + permutation
    nm2 = J.row_names(tab, rows.ids, J.decode_local(loc))
    back = np.empty(len(nm2), dtype=object)
    back[J.decode_perm(perm)] = nm2
    assert list(back) == list(names), g
    m2 = _pz2.Member(newblob)
    r2 = m2.decode(mode="canonical", threads=0, dtype="uint32")
    c2 = sp.csc_matrix((r2["data"], r2["indices"], r2["indptr"]), shape=(h["m"], h["n"]))
    legacy = c2[inv, :]
    assert (legacy != csc).nnz == 0, g
    assert list(m2.row_names()["pack_index"]) == rows.ids.tolist()
    names_bytes = h["sections"]["extra"][1]
    row = {
        "gsm": g,
        "rows": h["m"],
        "local_rows": len(rows.local),
        "old_member": len(blob),
        "old_names": names_bytes,
        "new_member": len(newblob),
        "local_list": len(loc),
        "legacy_perm": len(perm),
    }
    per.append(row)
    for k in (
        "old_member",
        "old_names",
        "new_member",
        "local_list",
        "legacy_perm",
        "rows",
        "local_rows",
    ):
        tot[k] += row[k]
out["per_sample"] = per
out["study_totals"] = dict(tot)
out["names_ms_one_sample"] = ms(lambda: J.row_names(tab, rows.ids, rows.local), reps=3)
# a reference operation measured in the same run, to scale the timings to an idle machine
ref = np.arange(1_000_000, dtype=np.uint32)
out["reference_cumsum_1M_u32"] = ms(lambda: np.cumsum(ref, dtype=np.uint32), reps=21)
print(json.dumps(out, indent=1))

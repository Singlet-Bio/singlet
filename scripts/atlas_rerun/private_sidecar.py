#!/usr/bin/env python3
"""Per-sample private sidecar: the withheld layers of a sample, kept on Clipper after the worker deletes its out dir.

Never public (nothing here reaches R2 or the `.s1`); written under the campaign's private location before the worker's clean-up.
What it holds (paths relative to the sample's out dir): everything `singlet._restricted.is_restricted_member` withholds (`private/`:
snp_ad / snp_dp / coverage / donor VCFs / raw chrM allele counts / haplotypes; `donor/` except donor_assignments.tsv; `mt/` except
mt_summary.tsv; ase_counts.tsv; snp_*, *.vcf) plus the layers the site does not offer: chrM heteroplasmy / variants / reference, ambient
(contamination, profile), doublet_scores, splice_psi / splice_events, sex_call and ancestry_call (untrimmed).

Size: each file is stored xz-compressed (a big MatrixMarket text, >= 50 MB, as delta-coded numeric arrays instead: much faster, smaller than xz-1 and lossless) (a `.gz` file is decompressed first and stored as xz: the *content* is kept, the gzip bytes are not),
and files with identical content are stored once (the snp_ad / snp_dp feature lists and the four barcode lists are copies of each other).
Nothing is dropped as "rebuildable": see SIDECAR.json `dup_of` for the only deduplication.

Layout: `<gsm>.private.tar` = SIDECAR.json + one `<rel>.xz` per distinct content. `verify()` re-reads the tar and checks every sha256;
`extract()` restores the files (gz files re-gzipped).
"""
import concurrent.futures
import gzip
import hashlib
import io
import json
import lzma
import os
import re
import tarfile

VERSION = 1
PRESET = 6
EXTRA_LAYERS = {
    "ambient_contamination.tsv", "ambient_contamination.json", "ambient_profile.tsv", "doublet_scores.tsv",
    "splice_psi.1pz", "splice_events.tsv", "sex_call.json", "ancestry_call.json",
    "mt_heteroplasmy.mtx.gz", "mt_heteroplasmy_barcodes.tsv.gz", "mt_heteroplasmy_features.tsv.gz", "mt_variants.tsv", "mt_reference.fa",
}
_DONOR_PUBLIC = {"donor_assignments.tsv"}
_MT_PUBLIC = {"mt_summary.tsv"}
_VCF = re.compile(r"\.(vcf|bcf)(\.gz)?$", re.I)


def is_withheld(rel):
    """The rule of python/singlet/_restricted.py::is_restricted_member (kept in step by hand; scripts/atlas_rerun does not import singlet)."""
    rel = rel.replace("\\", "/")
    if rel.startswith("private/"):
        return True
    if rel.startswith("donor/"):
        return rel[len("donor/"):] not in _DONOR_PUBLIC
    if rel.startswith("mt/"):
        return rel[len("mt/"):] not in _MT_PUBLIC
    if rel == "ase_counts.tsv":
        return True
    name = rel.rsplit("/", 1)[-1]
    return name.startswith("snp_") or bool(_VCF.search(name))


def select(out_dir):
    """-> sorted relative paths of the files that go into the sidecar."""
    rels = []
    for dp, _dn, fn in os.walk(out_dir):
        for f in fn:
            rel = os.path.relpath(os.path.join(dp, f), out_dir).replace("\\", "/")
            if rel.endswith("PRIVATE.sha256"):
                continue
            if is_withheld(rel) or rel in EXTRA_LAYERS:
                rels.append(rel)
    return sorted(rels)


def _content(path):
    """-> (bytes of the content, was_gz). A .gz file is read decompressed."""
    raw = open(path, "rb").read()
    if path.endswith(".gz"):
        try:
            return gzip.decompress(raw), True
        except OSError:
            pass
    return raw, False


MTX_MIN_BYTES = 50_000_000   # only the big text matrices get the numeric coding; xz-6 is kept for the rest (it is better on small text)


def _zc(b):
    try:
        import zstandard
        return b"Z" + zstandard.ZstdCompressor(level=10).compress(b)
    except ImportError:
        return b"X" + lzma.compress(b, preset=1)


def _zd(b):
    if b[:1] == b"Z":
        import zstandard
        return zstandard.ZstdDecompressor().decompress(b[1:], max_output_size=1 << 40)
    return lzma.decompress(b[1:])


def _digits(a):
    import numpy as np
    return np.floor(np.log10(np.maximum(a, 1))).astype(np.int64) + 1


def mtx_encode(data):
    """A MatrixMarket coordinate text of non-negative integers in column-major order -> a compact numeric coding, or None when `data` is not
    exactly the canonical text of such a matrix ("r c v\n" per entry, columns non-decreasing): the caller then falls back to xz."""
    import numpy as np
    pos, hdr_end = 0, 0
    while pos < len(data):
        e = data.find(bytes((10,)), pos)
        if e < 0:
            return None
        line = data[pos:e]
        pos = e + 1
        if line.startswith(b"%"):
            continue
        dims = line.split()
        if len(dims) != 3:
            return None
        hdr_end = pos
        break
    else:
        return None
    nnz = int(dims[2])
    body = data[hdr_end:]
    try:
        import pandas as pd
        import io
        arr = pd.read_csv(io.BytesIO(body), sep=" ", header=None, dtype=np.int64, engine="c").to_numpy()
    except Exception:  # noqa: BLE001
        return None
    if arr.shape != (nnz, 3) or arr.min(initial=0) < 0:
        return None
    r, c, v = arr[:, 0], arr[:, 1], arr[:, 2]
    if len(c) and (np.diff(c) < 0).any():
        return None
    if int((_digits(r) + _digits(c) + _digits(v)).sum()) + 3 * nnz != len(body) or not body.endswith(bytes((10,))):
        return None     # not the canonical one-space, one-newline text of these numbers
    if not body.startswith(_canon(arr[:6]).encode()) or not body.endswith(_canon(arr[-3:]).encode()):
        return None
    dc = np.diff(c, prepend=0)
    dr = np.diff(r, prepend=0)
    streams = []
    for a, signed in ((dc, False), (dr, True), (v, False)):
        lo, hi = int(a.min(initial=0)), int(a.max(initial=0))
        for t in ((np.int8, np.int16, np.int32, np.int64) if signed else (np.uint8, np.uint16, np.uint32, np.uint64)):
            if np.iinfo(t).min <= lo and hi <= np.iinfo(t).max:
                break
        streams.append(_zc(a.astype(t).tobytes()) if len(a) else b"")
        streams[-1] = (np.dtype(t).str.encode() + b";" + streams[-1])
    head = data[:hdr_end]
    import struct
    out = [struct.pack("<QQ", len(head), nnz), head]
    for st in streams:
        out += [struct.pack("<Q", len(st)), st]
    return b"".join(out)


def _canon(a):
    return "".join("%d %d %d" % tuple(x) + chr(10) for x in a.tolist())


def mtx_decode(blob):
    import struct
    import numpy as np
    hl, nnz = struct.unpack_from("<QQ", blob, 0)
    pos = 16
    head = blob[pos:pos + hl]
    pos += hl
    cols = []
    for _ in range(3):
        (n,) = struct.unpack_from("<Q", blob, pos)
        pos += 8
        st = blob[pos:pos + n]
        pos += n
        dt, _, comp = st.partition(b";")
        cols.append(np.frombuffer(_zd(comp), dtype=np.dtype(dt.decode())) if nnz else np.zeros(0, np.int64))
    dc, dr, v = (x.astype(np.int64) for x in cols)
    c, r = np.cumsum(dc), np.cumsum(dr)
    import io
    import pandas as pd
    buf = io.StringIO()
    pd.DataFrame({0: r, 1: c, 2: v}).to_csv(buf, sep=" ", header=False, index=False, lineterminator=chr(10))
    return head + buf.getvalue().encode()


def _pack(job):
    rel, path = job
    data, was_gz = _content(path)
    if len(data) >= MTX_MIN_BYTES and b"%%MatrixMarket" in data[:20]:
        blob = mtx_encode(data)
        if blob is not None:
            return rel, was_gz, hashlib.sha256(data).hexdigest(), len(data), blob, "mtx"
    return rel, was_gz, hashlib.sha256(data).hexdigest(), len(data), lzma.compress(data, preset=PRESET), "xz"


def build(out_dir, dest, gsm, gse=None, threads=4):
    """Write the sidecar tar for the sample whose engine output is out_dir. -> index dict (also stored as SIDECAR.json)."""
    rels = select(out_dir)
    with concurrent.futures.ThreadPoolExecutor(max(1, threads)) as ex:
        packed = list(ex.map(_pack, [(r, os.path.join(out_dir, r)) for r in rels]))
    files, seen = {}, {}
    tmp = dest + ".tmp"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with tarfile.open(tmp, "w") as tf:
        for rel, was_gz, sha, n, blob, codec in packed:
            ent = {"sha256": sha, "bytes": n, "gz": was_gz, "codec": codec}
            if sha in seen:
                ent["dup_of"] = seen[sha]
            else:
                seen[sha] = rel
                ent["stored"] = rel + (".xz" if codec == "xz" else ".mtxc")
                ent["xz_bytes"] = len(blob)
                ti = tarfile.TarInfo(rel + ".xz")
                ti.size = len(blob)
                ti.mtime = 0
                tf.addfile(ti, io.BytesIO(blob))
            files[rel] = ent
        idx = {"version": VERSION, "gsm": gsm, "gse": gse, "codec": "xz-preset%d; csc-delta+zstd for big mtx" % PRESET, "files": files,
               "note": ".gz files hold the decompressed content (re-gzipped on extract)"}
        b = json.dumps(idx, sort_keys=True, indent=1).encode()
        ti = tarfile.TarInfo("SIDECAR.json")
        ti.size = len(b)
        ti.mtime = 0
        tf.addfile(ti, io.BytesIO(b))
    os.replace(tmp, dest)
    return idx


def _unpack(ent, blob):
    return mtx_decode(blob) if ent.get("codec") == "mtx" else lzma.decompress(blob)


def read_index(path):
    with tarfile.open(path) as tf:
        return json.load(tf.extractfile("SIDECAR.json"))


def verify(path):
    """Re-read the tar and check every file's sha256. -> list of problems (empty = ok)."""
    bad = []
    with tarfile.open(path) as tf:
        idx = json.load(tf.extractfile("SIDECAR.json"))
        stored = {}
        for rel, ent in idx["files"].items():
            if "dup_of" in ent:
                if ent["dup_of"] not in idx["files"] or idx["files"][ent["dup_of"]]["sha256"] != ent["sha256"]:
                    bad.append("%s: bad dup_of" % rel)
                continue
            data = _unpack(ent, tf.extractfile(ent["stored"]).read())
            if hashlib.sha256(data).hexdigest() != ent["sha256"] or len(data) != ent["bytes"]:
                bad.append("%s: content mismatch" % rel)
            stored[rel] = True
    return bad


def extract(path, dest):
    """Restore the files under dest (a .gz file is written gzipped again). -> list of relative paths."""
    out = []
    with tarfile.open(path) as tf:
        idx = json.load(tf.extractfile("SIDECAR.json"))
        data_of = {}
        for rel, ent in idx["files"].items():
            src = ent.get("dup_of", rel)
            if src not in data_of:
                data_of[src] = _unpack(idx["files"][src], tf.extractfile(idx["files"][src]["stored"]).read())
            data = data_of[src]
            p = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(gzip.compress(data, 6, mtime=0) if ent["gz"] else data)
            out.append(rel)
    return out

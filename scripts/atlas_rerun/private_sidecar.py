#!/usr/bin/env python3
"""Per-sample private sidecar: the withheld layers of a sample, kept on Clipper after the worker deletes its out dir.

Never public (nothing here reaches R2 or the `.s1`); written under the campaign's private location before the worker's clean-up.
What it holds (paths relative to the sample's out dir): everything `singlet._restricted.is_restricted_member` withholds (`private/`:
snp_ad / snp_dp / coverage / donor VCFs / raw chrM allele counts / haplotypes; `donor/` except donor_assignments.tsv; `mt/` except
mt_summary.tsv; ase_counts.tsv; snp_*, *.vcf) plus the layers the site does not offer: chrM heteroplasmy / variants / reference, ambient
(contamination, profile), doublet_scores, splice_psi / splice_events, sex_call and ancestry_call (untrimmed).

Size: each file is stored xz-compressed (a `.gz` file is decompressed first and stored as xz: the *content* is kept, the gzip bytes are not),
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


def _pack(job):
    rel, path = job
    data, was_gz = _content(path)
    return rel, was_gz, hashlib.sha256(data).hexdigest(), len(data), lzma.compress(data, preset=PRESET)


def build(out_dir, dest, gsm, gse=None, threads=4):
    """Write the sidecar tar for the sample whose engine output is out_dir. -> index dict (also stored as SIDECAR.json)."""
    rels = select(out_dir)
    with concurrent.futures.ThreadPoolExecutor(max(1, threads)) as ex:
        packed = list(ex.map(_pack, [(r, os.path.join(out_dir, r)) for r in rels]))
    files, seen = {}, {}
    tmp = dest + ".tmp"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with tarfile.open(tmp, "w") as tf:
        for rel, was_gz, sha, n, blob in packed:
            ent = {"sha256": sha, "bytes": n, "gz": was_gz}
            if sha in seen:
                ent["dup_of"] = seen[sha]
            else:
                seen[sha] = rel
                ent["stored"] = rel + ".xz"
                ent["xz_bytes"] = len(blob)
                ti = tarfile.TarInfo(rel + ".xz")
                ti.size = len(blob)
                ti.mtime = 0
                tf.addfile(ti, io.BytesIO(blob))
            files[rel] = ent
        idx = {"version": VERSION, "gsm": gsm, "gse": gse, "codec": "xz-preset%d" % PRESET, "files": files,
               "note": ".gz files hold the decompressed content (re-gzipped on extract)"}
        b = json.dumps(idx, sort_keys=True, indent=1).encode()
        ti = tarfile.TarInfo("SIDECAR.json")
        ti.size = len(b)
        ti.mtime = 0
        tf.addfile(ti, io.BytesIO(b))
    os.replace(tmp, dest)
    return idx


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
            data = lzma.decompress(tf.extractfile(ent["stored"]).read())
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
                data_of[src] = lzma.decompress(tf.extractfile(idx["files"][src]["stored"]).read())
            data = data_of[src]
            p = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(gzip.compress(data, 6, mtime=0) if ent["gz"] else data)
            out.append(rel)
    return out

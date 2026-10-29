#!/usr/bin/env python
"""w4_vocab: range-read the row names (exon + intron), feature_vocab.json and summary.json of public .singlet bundles.

    python -P tools/w4_vocab_fetch.py GSE145370 GSE285936 ...          # first sample with both matrices
    python -P tools/w4_vocab_fetch.py --gsm GSE180661=GSM5467179 ...    # a specific sample
    python -P tools/w4_vocab_fetch.py --light GSE1 GSE2 ...             # only feature_vocab.json + summary.json (no names)
    python -P tools/w4_vocab_fetch.py --vocab-only GSE1 GSE2 ...        # only feature_vocab.json
    python -P tools/w4_vocab_fetch.py --local corpus/bundles/GSE145370.singlet   # same from a local bundle

Output (git-ignored) in corpus_w4_vocab/:
    <GSE>.feature_vocab.json, <GSE>.summary.json            (small members, verbatim)
    <GSE>.<GSM>.exon.meta / .intron.meta                    (the metadata block of the .1pz member, a zstd frame, verbatim:
                                                             TLV rownames + colnames + user kv)
    index.tsv                                                (one line per fetched member, see below)

Only the central directory, feature_vocab.json, summary.json and, per matrix, the 96 B header and the metadata block are
transferred (about 3-5 MB per matrix); the count payload is never read.  data.singlet.bio returns 403 to the default
Python-urllib User-Agent so one is set.
"""
import hashlib, io, json, os, struct, sys, time, urllib.request, zipfile
import zstandard

UA = {"User-Agent": "singlet-compression-study/0.1 (curl-compatible)"}
BASE = "https://data.singlet.bio/data/{gse}/{gse}.singlet"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "corpus_w4_vocab")
HDR = "<IHBBIIQBBHIIIIIQIIQII"      # first 80 bytes of the 96 B PZHeader
TAIL = 1 << 19


class Remote(io.RawIOBase):
    """Seekable read-only file over HTTP range requests with a cached tail (the zip central directory lives there)."""

    def __init__(self, url):
        self.url = url
        self.reqs = 0
        self.bytes = 0
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.tail_off = max(0, self.size - TAIL)
        self.tail = self.range(self.tail_off, self.size - self.tail_off)

    def range(self, off, n):
        if n <= 0:
            return b""
        end = min(self.size, off + n) - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": "bytes=%d-%d" % (off, end), **UA})
                with urllib.request.urlopen(req, timeout=600) as r:
                    d = r.read()
                self.reqs += 1
                self.bytes += len(d)
                if len(d) != end - off + 1:
                    raise IOError("short read %d != %d" % (len(d), end - off + 1))
                return d
            except Exception as e:  # transient network trouble
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.pos = off
        elif whence == io.SEEK_CUR: self.pos += off
        else: self.pos = self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if self.pos >= self.tail_off:
            d = self.tail[self.pos - self.tail_off:self.pos - self.tail_off + n]
        else:
            d = self.range(self.pos, n)
        self.pos += len(d)
        return d

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


class Local(io.FileIO):
    reqs = 0
    bytes = 0


def parse_tlv(raw):
    """metadata TLV -> (rownames, colnames, kv).  Tags: 1 rownames, 2 colnames, 3 user kv, 0 end."""
    rows, cols, kv = [], [], {}
    p = 0
    while p < len(raw):
        tag = raw[p]; p += 1
        if tag == 0:
            break
        (ln,) = struct.unpack_from("<I", raw, p); p += 4
        body = raw[p:p + ln]; p += ln
        if tag in (1, 2):
            parts = body.split(b"\0")
            assert parts[-1] == b"", "names block not NUL terminated"
            (rows if tag == 1 else cols).extend(x.decode("utf-8") for x in parts[:-1])
        elif tag == 3:
            parts = body.split(b"\0")[:-1]
            for i in range(0, len(parts), 2):
                kv[parts[i].decode()] = parts[i + 1].decode()
    return rows, cols, kv


def member_bytes(f, zi, off, n):
    """n bytes at offset `off` inside the (stored) member zi."""
    f.seek(zi.header_offset)
    lh = f.read(30)
    fn, ex = struct.unpack_from("<HH", lh, 26)
    data0 = zi.header_offset + 30 + fn + ex
    assert zi.compress_type == 0, "member is compressed in the zip; cannot range-read inside it"
    f.seek(data0 + off)
    return f.read(n)


def names_hash(lst):
    h = hashlib.sha256()
    for s in lst:
        h.update(s.encode()); h.update(b"\0")
    return h.hexdigest()


def read_meta(f, zi):
    hdr = member_bytes(f, zi, 0, 96)
    (magic, ver, vt, flags, m, n, nnz, pw, lvl, _p, nch, perm_z, ptr_z, chunk_cols, ff, meta_off, meta_z, colsums_z, t_off, t_z,
     t_ch) = struct.unpack(HDR, hdr[:80])
    assert magic == 0x5A315054, "bad magic"
    blk = member_bytes(f, zi, meta_off, meta_z)
    raw = zstandard.ZstdDecompressor().decompressobj().decompress(blk)
    return dict(m=m, n=n, nnz=nnz, meta_off=meta_off, meta_z=meta_z, size=zi.file_size), blk, raw


def process(gse, gsm_pref, light, local, idx, vocab_only=False):
    os.makedirs(OUT, exist_ok=True)
    if local:
        f = Local(local, "r"); src = local
    else:
        f = Remote(BASE.format(gse=gse)); src = BASE.format(gse=gse)
    zf = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16) if local else f)
    names = {i.filename: i for i in zf.infolist()}
    samples = sorted({k.split("/")[1] for k in names if k.startswith("samples/")})
    print("[%s] %d members, %d samples (%d requests so far)" % (gse, len(names), len(samples), f.reqs), flush=True)
    for small in ("feature_vocab.json",):
        if small in names:
            data = zf.read(small)
            with open(os.path.join(OUT, "%s.%s" % (gse, small)), "wb") as o:
                o.write(data)
    if vocab_only:
        print("[%s] done (vocab only): %d requests, %.2f MB transferred" % (gse, f.reqs, f.bytes / 1e6), flush=True)
        return "-"
    cand = [gsm_pref] if gsm_pref else samples
    done = None
    for gsm in cand:
        ex, it = "samples/%s/exon_counts.1pz" % gsm, "samples/%s/intron_counts.1pz" % gsm
        sj = "samples/%s/summary.json" % gsm
        if sj in names:
            sdat = zf.read(sj)
            with open(os.path.join(OUT, "%s.%s.summary.json" % (gse, gsm)), "wb") as o:
                o.write(sdat)
            summ = json.loads(sdat)
        else:
            summ = {}
        if light:
            done = gsm; break
        if ex not in names or it not in names:
            print("   %s: missing exon/intron member, skipping" % gsm); continue
        try:
            for kind, nm in (("exon", ex), ("intron", it)):
                info, blk, raw = read_meta(f, names[nm])
                rows, cols, kv = parse_tlv(raw)
                assert len(rows) == info["m"], (len(rows), info["m"])
                with open(os.path.join(OUT, "%s.%s.%s.meta" % (gse, gsm, kind)), "wb") as o:
                    o.write(blk)
                idx.append(dict(gse=gse, gsm=gsm, kind=kind, build=summ.get("reference_build", ""), m=info["m"], n=info["n"],
                                rows_sha256=names_hash(rows), cols_sha256=names_hash(cols), meta_z=info["meta_z"], raw_bytes=len(raw),
                                manifest_sha=summ.get("reference_manifest_sha256", ""), singlet_version=summ.get("singlet_version", ""),
                                git_sha=summ.get("git_sha", ""), protocol=summ.get("protocol_name", ""), species=summ.get("species", "")))
                print("   %s %-6s m=%d n=%d rows_sha=%s fetched so far %.1f MB" % (gsm, kind, info["m"], info["n"], idx[-1]["rows_sha256"][:12],
                                                                              f.bytes / 1e6), flush=True)
            done = gsm
            break
        except Exception as e:
            print("   %s failed: %r" % (gsm, e)); idx[:] = [r for r in idx if not (r["gse"] == gse and r["gsm"] == gsm)]
    print("[%s] done: sample=%s, %d requests, %.2f MB transferred" % (gse, done, f.reqs, f.bytes / 1e6), flush=True)
    return done


def main(argv):
    light = "--light" in argv
    vocab_only = "--vocab-only" in argv
    local = None
    gsm_map = {}
    gses = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--light", "--vocab-only"):
            pass
        elif a == "--local":
            local = argv[i + 1]; i += 1
        elif a == "--gsm":
            k, v = argv[i + 1].split("="); gsm_map[k] = v; i += 1
        elif not a.startswith("--"):
            gses.append(a)
        i += 1
    if local and not gses:
        gses = [os.path.basename(local).split(".")[0]]
    for a in list(gsm_map):
        if a not in gses:
            gses.append(a)
    idx_path = os.path.join(OUT, "index.tsv")
    old = []
    if os.path.exists(idx_path):
        lines = open(idx_path).read().splitlines()
        cols = lines[0].split("\t")
        old = [dict(zip(cols, l.split("\t"))) for l in lines[1:]]
    new = []
    for g in gses:
        process(g, gsm_map.get(g), light, local, new, vocab_only)
    keys = ["gse", "gsm", "kind", "build", "m", "n", "rows_sha256", "cols_sha256", "meta_z", "raw_bytes", "manifest_sha", "singlet_version",
            "git_sha", "protocol", "species"]
    merged = {(r["gse"], r["gsm"], r["kind"]): r for r in old}
    merged.update({(r["gse"], r["gsm"], r["kind"]): r for r in new})
    with open(idx_path, "w") as o:
        o.write("\t".join(keys) + "\n")
        for r in merged.values():
            o.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])

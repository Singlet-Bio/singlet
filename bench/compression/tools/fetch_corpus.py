#!/usr/bin/env python
"""Fetch selected members of remote .singlet (ZIP64) bundles over HTTP range requests.

Usage:
    python -P tools/fetch_corpus.py corpus.tsv corpus/            # all rows
    python -P tools/fetch_corpus.py corpus.tsv corpus/ --whole GSE145370   # also the full bundle

corpus.tsv columns: id  gse  gsm  note
For each row we pull  samples/<gsm>/{exon_counts.1pz,intron_counts.1pz,cell_calls.tsv,summary.json}
plus feature_vocab.json / manifest.json / study_meta.json once per study, into corpus/<id>/.
Only the central directory and the requested members are transferred.
"""
import io, os, sys, json, zipfile, urllib.request

UA = {"User-Agent": "singlet-compression-study/0.1 (curl-compatible)"}

BASE = "https://data.singlet.bio/data/{gse}/{gse}.singlet"
MEMBERS = ["exon_counts.1pz", "intron_counts.1pz", "cell_calls.tsv", "summary.json"]
STUDY = ["feature_vocab.json", "manifest.json", "study_meta.json"]


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (what zipfile needs)."""
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.bytes_fetched = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.pos = off
        elif whence == io.SEEK_CUR: self.pos += off
        else: self.pos = self.size + off
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n <= 0 or self.pos >= self.size: return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}", **UA})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        self.pos += len(data)
        self.bytes_fetched += len(data)
        return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def fetch_study(gse, rows, out_root):
    url = BASE.format(gse=gse)
    hf = HttpFile(url)
    zf = zipfile.ZipFile(io.BufferedReader(hf, buffer_size=1 << 20))
    names = set(zf.namelist())
    print(f"[{gse}] bundle {hf.size/1e6:.1f} MB, {len(names)} members", flush=True)
    for row in rows:
        d = os.path.join(out_root, row["id"]); os.makedirs(d, exist_ok=True)
        wanted = STUDY + [f"samples/{row['gsm']}/{m}" for m in MEMBERS]
        for w in wanted:
            if w not in names:
                print(f"  missing {w}", flush=True); continue
            dst = os.path.join(d, os.path.basename(w))
            if os.path.exists(dst) and os.path.getsize(dst) == zf.getinfo(w).file_size:
                continue
            with zf.open(w) as src, open(dst, "wb") as f:
                f.write(src.read())
            print(f"  {row['id']}: {os.path.basename(w)} {os.path.getsize(dst)/1e6:.2f} MB", flush=True)
        with open(os.path.join(d, "sample.json"), "w") as f:
            json.dump(row, f, indent=1)
    print(f"[{gse}] fetched {hf.bytes_fetched/1e6:.1f} MB over HTTP", flush=True)


def fetch_whole(gse, out_root):
    url = BASE.format(gse=gse)
    dst = os.path.join(out_root, "bundles", f"{gse}.singlet")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        print(f"[{gse}] whole bundle already present"); return
    print(f"[{gse}] downloading whole bundle -> {dst}", flush=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(dst + ".part", "wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk: break
            f.write(chunk)
    os.replace(dst + ".part", dst)
    print(f"[{gse}] done {os.path.getsize(dst)/1e6:.1f} MB", flush=True)


def main():
    tsv, out_root = sys.argv[1], sys.argv[2]
    whole = [a for a in sys.argv[4:] ] if len(sys.argv) > 3 and sys.argv[3] == "--whole" else []
    rows = []
    with open(tsv) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            if not line.strip() or line.startswith("#"): continue
            rows.append(dict(zip(hdr, line.rstrip("\n").split("\t"))))
    by_gse = {}
    for r in rows: by_gse.setdefault(r["gse"], []).append(r)
    for gse, rs in by_gse.items():
        fetch_study(gse, rs, out_root)
    for gse in whole:
        fetch_whole(gse, out_root)


if __name__ == "__main__":
    main()

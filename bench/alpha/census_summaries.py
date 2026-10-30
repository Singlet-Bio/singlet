"""Read summary.json of every Research Alpha public sample straight out of the published bundles (runs on Clipper).

    python -P census_summaries.py GSM_LIST_TSV OUT_TSV --shard K --of N [--threads 16]

Per study: the zip central directory and then each wanted summary.json member are read by exact HTTP ranges
(no read-ahead), so no request overlaps a member data.singlet.bio refuses to serve. Shard K of N takes every N-th study.
"""
import concurrent.futures as cf
import csv
import io
import json
import sys
import time
import urllib.request
import zipfile

UA = {"User-Agent": "singlet-alpha-bench/1.0 (+https://singlet.bio)"}
gl, out = sys.argv[1:3]
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d  # noqa: E731
K, N, NT = arg("--shard", 0), arg("--of", 1), arg("--threads", 16)
want = {}
for r in csv.DictReader(open(gl, encoding="utf-8"), delimiter="\t"):
    want.setdefault(r["gse_id"], set()).add(r["gsm_id"])
studies = sorted(want)[K::N]
KEYS = ["n_input_reads", "total_reads", "mapping_rate", "n_cells_called", "median_umi_per_cell", "median_genes_per_cell",
        "git_sha", "singlet_version", "protocol_name", "reference_build", "species", "wall_seconds", "peak_rss_gb",
        "fraction_reads_in_cells"]


def http(url, a, b):
    for i in range(6):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as x:
                d = x.read()
            if len(d) == b - a + 1:
                return d
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 416):
                raise
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2 * (i + 1))
    raise IOError("range read failed")


class Exact(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if (n is None or n < 0) else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        d = http(self.url, self.pos, self.pos + n - 1)
        self.pos += n
        return d

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def study(g):
    rows = []
    try:
        zf = zipfile.ZipFile(Exact(f"https://data.singlet.bio/data/{g}/{g}.singlet"))
        names = set(zf.namelist())
    except Exception as e:  # noqa: BLE001
        return [dict(gse=g, gsm=s, error=f"bundle: {type(e).__name__}: {e}") for s in sorted(want[g])]
    for s in sorted(want[g]):
        p = f"samples/{s}/summary.json"
        if p not in names:
            rows.append(dict(gse=g, gsm=s, error="no summary.json"))
            continue
        try:
            js = json.loads(zf.read(p))
            rows.append(dict(gse=g, gsm=s, **{k: js.get(k) for k in KEYS}))
        except Exception as e:  # noqa: BLE001
            rows.append(dict(gse=g, gsm=s, error=f"{type(e).__name__}: {e}"))
    return rows


with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, ["gse", "gsm"] + KEYS + ["error"], delimiter="\t", lineterminator="\n")
    w.writeheader()
    with cf.ThreadPoolExecutor(NT) as ex:
        for i, rows in enumerate(ex.map(study, studies)):
            w.writerows(rows)
            if i % 200 == 0:
                print(K, i, len(studies), flush=True)
print("done", K, len(studies))

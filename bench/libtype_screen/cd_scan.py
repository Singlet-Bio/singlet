# SPDX-License-Identifier: MIT
"""Per-sample member sizes from every legacy bundle's ZIP central directory (range reads only)."""

import io
import os
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

# working directory for caches and outputs (default: next to this script)
HERE = os.environ.get("LIBTYPE_WORK", os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "singlet-libtype-screen/1.0 (+https://singlet.bio)"}


class RF(io.RawIOBase):
    def __init__(s, u):
        s.u, s.p = u, 0
        with urllib.request.urlopen(
            urllib.request.Request(u, method="HEAD", headers=UA), timeout=60
        ) as r:
            s.n = int(r.headers["Content-Length"])

    def readable(s):
        return True

    def seekable(s):
        return True

    def tell(s):
        return s.p

    def seek(s, o, w=0):
        s.p = o if w == 0 else s.p + o if w == 1 else s.n + o
        return s.p

    def readinto(s, b):
        if s.p >= s.n:
            return 0
        e = min(s.n, s.p + len(b)) - 1
        h = dict(UA, Range=f"bytes={s.p}-{e}")
        with urllib.request.urlopen(urllib.request.Request(s.u, headers=h), timeout=120) as r:
            d = r.read()
        b[: len(d)] = d
        s.p += len(d)
        return len(d)


def scan(gse):
    for attempt in range(3):
        try:
            z = zipfile.ZipFile(
                io.BufferedReader(RF(f"https://data.singlet.bio/data/{gse}/{gse}.singlet"), 1 << 18)
            )
            rows = []
            for i in z.infolist():
                p = i.filename.split("/")
                if len(p) >= 3 and p[0] == "samples":
                    rows.append((gse, p[1], "/".join(p[2:]), i.compress_size))
            return gse, rows, None
        except Exception as e:  # noqa: BLE001
            err = repr(e)
    return gse, [], err


if __name__ == "__main__":
    gses = (
        [ln.strip() for ln in open(sys.argv[1]) if ln.strip()]
        if len(sys.argv) > 1
        else sorted(pd.read_parquet(os.path.join(HERE, "d1_qc.parquet")).gse_id.unique())
    )
    out, errs = [], []
    with ThreadPoolExecutor(12) as ex:
        for k, (gse, rows, err) in enumerate(ex.map(scan, gses)):
            out += rows
            if err:
                errs.append((gse, err))
            if k % 500 == 0:
                print(k, len(out), len(errs), flush=True)
    pd.DataFrame(out, columns=["gse_id", "gsm_id", "member", "bytes"]).to_parquet(
        os.path.join(HERE, "members.parquet")
    )
    pd.DataFrame(errs, columns=["gse_id", "error"]).to_csv(
        os.path.join(HERE, "members_errors.csv"), index=False
    )
    print("done", len(out), len(errs), flush=True)

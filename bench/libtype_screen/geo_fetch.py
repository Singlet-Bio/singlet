# SPDX-License-Identifier: MIT
"""Fetch the GEO family SOFT file of every atlas series (cache: <work>/soft/<GSE>.soft.gz) from NCBI's static FTP-over-HTTPS tree."""

import gzip
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

# working directory for caches and outputs (default: next to this script)
HERE = os.environ.get("LIBTYPE_WORK", os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "soft")
os.makedirs(OUT, exist_ok=True)


def url(g):
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{g[:-3]}nnn/{g}/soft/{g}_family.soft.gz"


lock = threading.Lock()
gate = [0.0]


def fetch(gse):
    path = os.path.join(OUT, gse + ".soft.gz")
    if os.path.exists(path):
        return gse, "cached"
    for attempt in range(6):
        with lock:  # <= ~20 requests/s overall
            wait = gate[0] - time.time()
            if wait > 0:
                time.sleep(wait)
            gate[0] = max(gate[0], time.time()) + 0.05
        try:
            req = urllib.request.Request(
                url(gse),
                headers={"User-Agent": "singlet-libtype-screen/1.0 (+https://singlet.bio)"},
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if gzip.decompress(data).find(b"^SAMPLE") < 0:
                raise RuntimeError("no samples in response")
            with open(path + ".tmp", "wb") as f:
                f.write(data)
            os.replace(path + ".tmp", path)
            return gse, "ok"
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(5 * (attempt + 1))
    return gse, "FAIL " + err


if __name__ == "__main__":
    gses = sorted(pd.read_parquet(os.path.join(HERE, "d1_qc.parquet")).gse_id.unique())
    done = 0
    with ThreadPoolExecutor(8) as ex:
        for gse, st in ex.map(fetch, gses):
            done += 1
            if st != "cached" and (st != "ok" or done % 200 == 0):
                print(done, gse, st, flush=True)
    print("finished", done, flush=True)

# SPDX-License-Identifier: MIT
"""Range-chunked download of GEO sample supplementary files (the NCBI mirror closes long responses early from Clipper).

    python -P geo_fetch.py OUTDIR URL [URL ...]
"""
import os
import sys
import time
import urllib.request

CH = 256 * 1024


def size(url):
    for _ in range(5):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}), timeout=60)
            return int(r.headers["Content-Length"])
        except Exception as e:  # noqa: BLE001
            print("HEAD", e, flush=True)
            time.sleep(3)
    raise SystemExit("no size for " + url)


def fetch(url, out):
    n = size(url)
    have = os.path.getsize(out) if os.path.exists(out) else 0
    tries = 0
    with open(out, "ab") as fh:
        while have < n and tries < 400:
            a, b = have, min(have + CH, n) - 1
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={a}-{b}"})
                d = urllib.request.urlopen(req, timeout=60).read()
                if not d:
                    raise IOError("empty")
                fh.write(d)
                have += len(d)
            except Exception as e:  # noqa: BLE001
                tries += 1
                print("retry", a, e, flush=True)
                time.sleep(1)
    print(out, have, "of", n, "OK" if have == n else "INCOMPLETE", flush=True)


if __name__ == "__main__":
    os.makedirs(sys.argv[1], exist_ok=True)
    for u in sys.argv[2:]:
        fetch(u, os.path.join(sys.argv[1], u.rsplit("/", 1)[1]))

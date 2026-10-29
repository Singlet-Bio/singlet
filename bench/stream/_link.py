# SPDX-License-Identifier: MIT
"""The machine's own download capacity, measured next to each benchmark: four parallel 25 MB downloads from Cloudflare's speed
test endpoint (the same network as data.singlet.bio), so a streaming rate can be read against what the link allows."""

from __future__ import annotations

import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://speed.cloudflare.com/__down?bytes=25000000"


def link_speed(connections: int = 4) -> dict:
    def one(_):
        req = urllib.request.Request(URL, headers={"User-Agent": "singlet-bench/1"})
        with urllib.request.urlopen(req, timeout=120) as r:
            n = 0
            while True:
                b = r.read(1 << 20)
                if not b:
                    return n
                n += len(b)

    t = time.perf_counter()
    try:
        with ThreadPoolExecutor(connections) as ex:
            total = sum(ex.map(one, range(connections)))
    except OSError as e:  # the benchmark still runs; the report says why there is no link figure
        return {"error": repr(e)}
    sec = time.perf_counter() - t
    return {
        "source": URL,
        "connections": connections,
        "bytes": total,
        "seconds": round(sec, 2),
        "mb_per_s": round(total / 1e6 / sec, 2),
    }

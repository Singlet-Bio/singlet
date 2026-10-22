"""m_w4: measure the read path (legacy bundle vs v2) in fresh processes: wall time, peak memory (process RSS and tracemalloc), requests, bytes.

Parent mode (default) spawns one child per configuration so memory is not shared between runs:

    python -P tools/m_w4_measure.py --v2 results/m_w4_v2/native14 --bundle corpus/bundles/GSE145370.singlet --reps 3 \
        --out results/m_w4_measure.tsv

Child mode (--child KIND) loads once and prints one JSON line. KINDs:
    legacy_study     SingletBundle.open(bundle).to_anndata(layers=True)       (what singlet.load(path) does)
    legacy_sample    the legacy per-sample work: _load_gsm_gene_counts + the two layer parts (what to_anndata does per GSM)
    v2_study_local   open_study(<dir>) + to_anndata()       (objects read from disk, memory mapped)
    v2_sample_local  open_study(<dir>) + to_anndata([GSM])
    v2_study_http    same over loopback HTTP with Range requests (a separate server process counts requests and bytes)
    v2_sample_http
    v2_cells_http    cells("n_umi_exon >= <thr>") on one sample, partial plan
All children report time after the imports (cold imports are not part of a load), RSS growth over the post-import baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GSM = "GSM4317426"


def _rss_peak():
    import psutil

    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss)


def child(args):
    server_box = []
    try:
        _child(args, server_box)
    finally:
        for sv in server_box:
            sv.terminate()


def _child(args, server_box):
    sys.path.insert(0, os.path.join(ROOT, "python"))
    sys.path.insert(0, os.path.join(ROOT, "tests", "python"))
    import psutil  # noqa: F401
    import tracemalloc

    import anndata  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import scipy.sparse  # noqa: F401

    import singlet  # noqa: F401

    os.environ["SINGLET_CACHE_DIR"] = tempfile.mkdtemp(prefix="m_w4_cache_")
    kind = args.child
    server = None
    base = None
    if kind.endswith("_http"):
        import json as _j

        # mkstemp creates the files atomically (mktemp only invents a name that another process could claim first)
        fd, port_file = tempfile.mkstemp(prefix="m_w4_port_")
        os.close(fd)
        fd, log_file = tempfile.mkstemp(prefix="m_w4_log_")
        os.close(fd)
        server = subprocess.Popen([sys.executable, "-P", os.path.join(HERE, "m_w4_serve.py"), args.v2, port_file, log_file])
        server_box.append(server)
        for _ in range(200):
            if os.path.exists(port_file) and os.path.getsize(port_file) > 0:
                break
            time.sleep(0.05)
        base = "http://127.0.0.1:%s" % open(port_file).read().strip()
    elif kind.startswith("v2_"):
        base = args.v2
    baseline = psutil.Process().memory_info().rss
    out = {"kind": kind, "baseline_rss_mb": baseline / 2**20}
    if args.tracemalloc:
        tracemalloc.start()
    t0 = time.perf_counter()
    if kind == "legacy_study":
        from singlet.bundle import SingletBundle

        ad = SingletBundle.open(args.bundle).to_anndata(verbose=False)
    elif kind == "legacy_sample":
        from singlet.bundle import SingletBundle

        b = SingletBundle.open(args.bundle)
        mat, bcs = b._load_gsm_gene_counts(GSM)
        b._gene_level_part(GSM, "exon", bcs)
        b._gene_level_part(GSM, "intron", bcs)
        ad = None
        out["shape"] = list(mat.shape)
    else:
        from singlet.v2 import client

        st = client.open_study("GSE145370", base=base)
        t_open = time.perf_counter() - t0
        out["open_s"] = t_open
        if kind in ("v2_study_local", "v2_study_http"):
            ad = st.to_anndata()
        elif kind in ("v2_sample_local", "v2_sample_http"):
            ad = st.to_anndata([GSM])
        elif kind == "v2_cells_http":
            ad = st.cells("n_umi_exon >= 300", samples=[GSM], whole_max=1)
        elif kind == "v2_slice1000_http":      # the 1,000 largest cells of one sample, two-phase plan
            ad = st.cells(None, samples=[GSM], max_cells=1000, whole_max=1)
        elif kind == "v2_slice1000_whole_http":  # the same slice with the default plan (the sample is fetched whole)
            ad = st.cells(None, samples=[GSM], max_cells=1000)
        else:
            raise SystemExit("unknown kind " + kind)
    out["wall_s"] = time.perf_counter() - t0
    if args.tracemalloc:
        cur, peak = tracemalloc.get_traced_memory()
        out["tracemalloc_peak_mb"] = peak / 2**20
        tracemalloc.stop()
    out["peak_rss_mb"] = _rss_peak() / 2**20
    out["rss_growth_mb"] = out["peak_rss_mb"] - out["baseline_rss_mb"]
    if ad is not None:
        out["shape"] = list(ad.shape)
        out["nnz"] = int(getattr(ad.X, "nnz", 0))
        out["layers"] = sorted(ad.layers.keys())
    if server is not None:
        server.terminate()
        server.wait(timeout=10)
        rows = [json.loads(x) for x in open(log_file).read().splitlines() if x.strip()]
        out["requests"] = len(rows)
        out["bytes"] = sum(r["bytes"] for r in rows)
        out["by_kind"] = {}
        for r in rows:
            k = "manifest" if "/m/" in r["path"] else ("packs" if "/ref/" in r["path"] else "object")
            d = out["by_kind"].setdefault(k, [0, 0])
            d[0] += 1
            d[1] += r["bytes"]
    print("RESULT " + json.dumps(out))


LOCK = os.path.join(ROOT, "bench", "compression", "results", ".timing_lock")


def _lock():
    """results/.timing_lock protocol: mkdir is atomic; wait while another agent holds it (a lock older than 20 min is stale)."""
    for _ in range(3000):
        try:
            os.mkdir(LOCK)
            return
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(LOCK) > 1200:
                    os.rmdir(LOCK)
                    continue
            except OSError:
                pass
            time.sleep(2)
    raise SystemExit("timing lock busy")


def _unlock():
    try:
        os.rmdir(LOCK)
    except OSError:
        pass


def _cpu():
    """'clock MHz/load %' from Windows CIM, '' when unavailable."""
    try:
        cmd = "(Get-CimInstance Win32_Processor | Select-Object -First 1 | ForEach-Object { '{0}/{1}' -f $_.CurrentClockSpeed, $_.LoadPercentage })"
        return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def parent(args):
    kinds = args.kinds.split(",")
    rows = []
    for kind in kinds:
        for rep in range(args.reps):
            for tm in ([False, True] if (args.tracemalloc_pass and rep == 0) else [False]):
                cmd = [sys.executable, "-P", os.path.abspath(__file__), "--child", kind, "--v2", args.v2, "--bundle", args.bundle]
                if tm:
                    cmd.append("--tracemalloc")
                load0 = _cpu()
                _lock()
                try:
                    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                finally:
                    _unlock()
                load1 = _cpu()
                line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
                if not line:
                    print("FAILED", kind, p.stderr[-2000:], file=sys.stderr)
                    continue
                r = json.loads(line[0][7:])
                r["rep"] = rep
                r["cpu_before"], r["cpu_after"] = load0, load1
                r["tracemalloc"] = tm
                rows.append(r)
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "layers"}), flush=True)
    if args.out:
        keys = ["kind", "rep", "tracemalloc", "cpu_before", "cpu_after", "wall_s", "open_s", "peak_rss_mb", "rss_growth_mb", "tracemalloc_peak_mb", "requests", "bytes", "shape", "nnz"]
        with open(args.out, "w") as fh:
            fh.write("\t".join(keys) + "\n")
            for r in rows:
                fh.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child")
    ap.add_argument("--v2", default=os.path.join(ROOT, "bench", "compression", "results", "m_w4_v2", "native14"))
    ap.add_argument("--bundle", default=os.path.join(ROOT, "bench", "compression", "corpus", "bundles", "GSE145370.singlet"))
    ap.add_argument("--kinds", default="legacy_sample,legacy_study,v2_sample_local,v2_study_local,v2_sample_http,v2_study_http,v2_cells_http,v2_slice1000_http,v2_slice1000_whole_http")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tracemalloc", action="store_true")
    ap.add_argument("--tracemalloc-pass", action="store_true", help="parent: one extra run per kind under tracemalloc")
    ap.add_argument("--out")
    a = ap.parse_args()
    child(a) if a.child else parent(a)

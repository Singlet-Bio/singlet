"""m_w4: network-time model for the v2 read path (requests and bytes are exact; times are a MODEL).

Requests and bytes come from the real manifest (`Study.request_plan`) and, for slices, from the partial plan driven against a local
Range server (see tools/m_w4_measure.py kinds v2_slice1000_http / v2_cells_http). Times use the coordinator's constants from
notes/w4_reader.md section 2: TTFB 0.22 s per request, one stream 32 MB/s, handshake 2 RTT (RTT 50 ms), and 8 parallel streams only
18 MB/s aggregate on that laptop (so concurrency is modelled as 1 stream at 32 MB/s with the requests of one phase overlapping their
TTFB, and as a pessimistic 18 MB/s aggregate).

    python -P tools/m_w4_netmodel.py <v2 root> <legacy bundle size in bytes> [slice_requests slice_bytes slice_phases]
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))

TTFB, RATE, AGG8, HANDSHAKE = 0.22, 32e6, 18e6, 0.10


def t(phases, nbytes, rate=RATE):
    """Sequential phases (each costs one TTFB; requests inside a phase overlap) plus transfer time."""
    return HANDSHAKE + phases * TTFB + nbytes / rate


def main(v2root, legacy_bytes, slice_info=None):
    os.environ["SINGLET_CACHE_DIR"] = tempfile.mkdtemp(prefix="m_w4_net_")
    from singlet.v2 import client

    st = client.open_study("GSE145370", base=v2root)
    man_bytes = os.path.getsize(os.path.join(v2root, "v2", "m", "GSE145370.json"))
    plan_all = st.request_plan(include_manifest=False)
    gsm = "GSM4317426"
    one = st.request_plan([gsm], include_manifest=False)
    rows = []
    # legacy: GSM -> API lookup (1-2 requests, tiny) then the whole bundle on one stream, then verification passes (CPU, not modelled here)
    rows.append(("legacy one sample (whole bundle)", 2, legacy_bytes, t(2, legacy_bytes)))
    rows.append(("legacy study (whole bundle)", 1, legacy_bytes, t(1, legacy_bytes)))
    # v2: manifest (phase 1) then the sample members (phase 2)
    rows.append(("v2 one sample", 1 + one["requests"], man_bytes + one["bytes"], t(2, man_bytes + one["bytes"])))
    rows.append(("v2 one sample, manifest cached (304)", 1 + one["requests"], one["bytes"], t(2, one["bytes"])))
    rows.append(("v2 study, 1 stream", 1 + plan_all["requests"], man_bytes + plan_all["bytes"], t(1 + 1, man_bytes + plan_all["bytes"])))
    # with 4 connections the 14 requests overlap their TTFB: ceil(14/4) waves of 0.22 s, but at most 18 MB/s aggregate on that link
    waves = -(-plan_all["requests"] // 4)
    rows.append(("v2 study, 4 connections (TTFB overlapped, 18 MB/s aggregate)", 1 + plan_all["requests"], man_bytes + plan_all["bytes"],
                 HANDSHAKE + TTFB + waves * TTFB + (man_bytes + plan_all["bytes"]) / AGG8))
    if slice_info:
        rq, by, ph = slice_info
        rows.append(("v2 1,000-cell slice of one sample (two-phase plan)", 1 + rq, man_bytes + by, t(1 + ph, man_bytes + by)))
    print("%-62s %9s %14s %9s" % ("scenario", "requests", "bytes", "model s"))
    for name, rq, by, sec in rows:
        print("%-62s %9d %14s %9.2f" % (name, rq, format(int(by), ","), sec))
    print("\nper sample (v2 whole-sample plan):")
    for p in plan_all["samples"]:
        print("  %s requests %d bytes %s (object %s)" % (p["gsm"], p["requests"], format(p["bytes"], ","), format(p["object_bytes"], ",")))
    json.dump({"rows": rows, "man_bytes": man_bytes}, open(os.path.join(ROOT, "bench", "compression", "results", "m_w4_netmodel.json"), "w"))


if __name__ == "__main__":
    v2root = sys.argv[1]
    legacy = int(sys.argv[2])
    sl = tuple(int(x) for x in sys.argv[3:6]) if len(sys.argv) >= 6 else None
    main(v2root, legacy, sl)

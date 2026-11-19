#!/usr/bin/env python3
"""t2-fallzone census: per sample of the named campaign directories, the engine-level evidence the runner kept.
Reads campaign_provenance.json (a fixed list of fields), ROUTE.json (kind, reason), run.log, engine.gpu.log.gz and the item's own
out/star_Log.out (regex hits on the engine's lines only). Never opens any other file of out/ (no private/, no HLA file).
Lists samples/<shard>/<key> of each named campaign directory (no deeper listing).
usage: t2-fallzone-census.py CAMPAIGN_DIR [...]   -> TSV on stdout"""
import gzip, json, os, re, sys

RX = re.compile(r"FELL BACK|level 0 \(requested|level \d \(requested|E8 records not built|E17 budget|read chunks|fallback")
LV = re.compile(r"E19: level (\d) reached \(requested (\d)\)|level (\d) \(requested (\d)\)")
DS = re.compile(r"engine_gpu: (\d+) reads, (\d+) distinct")
NB = re.compile(r"E8 records not built on the device \(([^\n]{0,160})")
BU = re.compile(r"E17 budget ([\d.]+) GB \([^;]*; device ([\d.]+) GB, ([\d.]+) GB in use at the start\): mode (\w+)")
RCX = re.compile(r"P3-MEM-03 reads phase in (\d+) read chunks")
FB = re.compile(r"FELL BACK")


def tail(path, n=8 << 20):
    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", errors="replace") as f:
                return f.read(n)
        with open(path, errors="replace") as f:
            return f.read(n)
    except OSError:
        return ""


def j(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def starlog(d):
    """level, reads, distinct, fall-back reason, read chunks, budget, mode/device/used from the item's out/star_Log.out."""
    t = tail(os.path.join(d, "out", "star_Log.out"))
    if not t:
        return ["", "", "", "", "", "", ""]
    lv = ""
    for m in LV.finditer(t):
        lv = m.group(1) or m.group(3) or lv
    ds = DS.search(t); nb = NB.search(t); bu = BU.search(t); rc = RCX.search(t)
    return [lv, ds.group(1) if ds else "", ds.group(2) if ds else "",
            ("FELL BACK" if FB.search(t) else "") + ((" " + nb.group(1)) if nb else ""),
            rc.group(1) if rc else "0", bu.group(1) if bu else "", (bu.group(4) + "/dev" + bu.group(2) + "/used" + bu.group(3)) if bu else ""]


print("\t".join(["campaign", "shard", "key", "device", "lane", "pairs", "wall_s", "star_s", "workers", "slots", "gpu_mem_gb", "fallbacks_flag",
                 "dev_peak_mib", "runlog_bytes", "englog_gz_bytes", "regex_hits", "route_reasons", "written",
                 "level", "n_reads", "n_distinct", "fallback_reason", "read_chunks", "budget_gb", "mode_dev_used"]))
for cdir in sys.argv[1:]:
    cname = os.path.basename(cdir.rstrip("/"))
    sroot = os.path.join(cdir, "samples")
    try:
        shards = sorted(os.listdir(sroot))
    except OSError:
        continue
    for sh in shards:
        try:
            keys = sorted(os.listdir(os.path.join(sroot, sh)))
        except OSError:
            continue
        for k in keys:
            d = os.path.join(sroot, sh, k)
            p = j(os.path.join(d, "campaign_provenance.json")) or {}
            r = p.get("route") or {}; ru = p.get("run") or {}; po = p.get("pool") or {}; el = p.get("engine_log") or {}
            su = el.get("summary") or {}; fo = el.get("found") or {}
            hits = set()
            for fn in ("run.log", "engine.gpu.log.gz"):
                for m in RX.finditer(tail(os.path.join(d, fn))):
                    hits.add(m.group(0))
            rt = j(os.path.join(d, "ROUTE.json")) or {}
            reasons = ";".join(sorted({str(f.get("kind", "")) + ":" + str(f.get("reason") or f.get("memory_rule") or "")[:80]
                                       for f in rt.get("failures", []) if isinstance(f, dict)}))

            def sz(fn):
                try:
                    return os.path.getsize(os.path.join(d, fn))
                except OSError:
                    return -1
            row = [cname, sh, k, r.get("device", ""), r.get("lane", ""), ru.get("pairs", ""), ru.get("wall_s", ""), su.get("star_seconds", ""),
                   po.get("workers", ""), po.get("slots", ""), po.get("gpu_mem_gb", ""), ",".join(fo.get("fallbacks") or []),
                   (el.get("measured_peaks") or {}).get("device_peak_mib", ""), sz("run.log"), sz("engine.gpu.log.gz"), "|".join(sorted(hits)), reasons,
                   p.get("written", "")] + starlog(d)
            print("\t".join(str(x) for x in row))

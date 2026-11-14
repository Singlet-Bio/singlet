#!/usr/bin/env python3
# nsys_an.py <prof.sqlite> : slot-hold analysis of a node run profiled by nsys (SINGLET_NVTX=1 ranges). Per slot hold (NVTX "device lock: held*"):
# label (from the "phase: X" range inside it), hold seconds, GPU busy (union of kernels+copies+memsets of the process, or of the non-host server),
# idle seconds and what the host was doing in the gaps, kernels by hold label, copies, malloc/free/sync counts.
import sqlite3, sys, collections
import numpy as np

db = sys.argv[1]
c = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
S = {i: v for i, v in c.execute("select id,value from StringIds")}
def q(sql): return c.execute(sql).fetchall()

# ---------------- NVTX ranges
R = []   # pid, start, end, text
RT5 = []   # the same with the thread id
for st, en, txt, tid, tx in q("select start,end,text,globalTid,textId from NVTX_EVENTS where eventType in (59,60) and end is not null"):
    R.append((tid >> 24, st, en, txt if txt else S.get(tx, "")))
    RT5.append((tid >> 24, st, en, txt if txt else S.get(tx, ""), tid))
print("nvtx ranges", len(R), "pids", sorted(set(r[0] for r in R)))
holds = [r for r in RT5 if r[3].startswith("device lock: held")]
waits = [r for r in R if r[3].startswith("device lock: wait")]
phases = [r for r in R if r[3].startswith("phase: ")]
print("holds", len(holds), "waits", len(waits), "phase ranges", len(phases))

# ---------------- GPU work
def arr(sql, ncol):
    rows = q(sql)
    return np.array(rows, dtype=np.int64).reshape(-1, ncol) if rows else np.zeros((0, ncol), dtype=np.int64)
K = arr("select globalPid>>24,start,end,deviceId,shortName from CUPTI_ACTIVITY_KIND_KERNEL", 5)
M = arr("select globalPid>>24,start,end,deviceId,copyKind,bytes,srcKind,dstKind from CUPTI_ACTIVITY_KIND_MEMCPY", 8)
Z = arr("select globalPid>>24,start,end,deviceId from CUPTI_ACTIVITY_KIND_MEMSET", 4)
print("kernels", len(K), "memcpy", len(M), "memset", len(Z))
t_lo = min(K[:, 1].min() if len(K) else 1 << 62, min(r[1] for r in R) if R else 1 << 62)
t_hi = max(K[:, 2].max() if len(K) else 0, max(r[2] for r in R) if R else 0)
print("trace span %.1f s" % ((t_hi - t_lo) / 1e9))

class Cover:
    """union of intervals; F(t) = covered time before t"""
    def __init__(self, s, e):
        if len(s) == 0: self.S = self.E = self.C = np.zeros(0); return
        o = np.argsort(s, kind="stable"); s = s[o]; e = e[o]
        ce = np.maximum.accumulate(e)
        new = np.concatenate(([True], s[1:] > ce[:-1]))
        idx = np.flatnonzero(new)
        self.S = s[idx]; self.E = np.concatenate((ce[idx[1:] - 1], [ce[-1]]))
        self.C = np.concatenate(([0], np.cumsum(self.E - self.S)))
    def F(self, t):
        if len(self.S) == 0: return 0
        i = np.searchsorted(self.S, t, side="right")   # intervals started at or before t
        if i == 0: return 0
        tot = self.C[i - 1] + min(t, self.E[i - 1]) - self.S[i - 1]
        return tot
    def cov(self, a, b): return self.F(b) - self.F(a)
    def gaps(self, a, b):
        """(start,end) of uncovered stretches inside [a,b]"""
        out = []; cur = a
        i = np.searchsorted(self.E, a, side="right")
        while i < len(self.S) and self.S[i] < b:
            if self.S[i] > cur: out.append((cur, self.S[i]))
            cur = max(cur, self.E[i]); i += 1
        if cur < b: out.append((cur, b))
        return out

def per_pid(*tabs):
    d = {}
    pids = set()
    for t in tabs: pids |= set(np.unique(t[:, 0]).tolist())
    for p in pids:
        ss = [t[t[:, 0] == p][:, 1] for t in tabs]; ee = [t[t[:, 0] == p][:, 2] for t in tabs]
        d[p] = Cover(np.concatenate(ss), np.concatenate(ee))
    return d
cov_all = per_pid(K, M, Z)
cov_k = per_pid(K)
cov_m = per_pid(M)

_nhid = [i for i, v in S.items() if v in ("k_probe", "k_first")]
NHP = sorted(set(int(p) for p in K[np.isin(K[:, 4], _nhid)][:, 0])) if _nhid else []
print("non-host server pids (kernels k_probe/k_first):", NHP)
# ---------------- label each hold
def overlap(a0, a1, b0, b1): return max(0, min(a1, b1) - max(a0, b0))
rows = []
for (pid, h0, h1, txt, htid) in holds:
    best = {}; lab = None; work = pid
    # 1. the thread that took the hold is inside the non-host screen: the kernels run in the pool's server process
    if any(t2 == htid and (t.startswith("main: non-host screen") or t.startswith("nonhost")) and overlap(h0, h1, s, e) > 0 for (p2, s, e, t, t2) in RT5):
        for (p2, s, e, t) in R:
            if p2 != pid and t.startswith("phase: nonhost") and overlap(h0, h1, s, e) > 0.3 * (h1 - h0):
                lab = "nonhost"; work = p2; break
        if lab is None: lab = "nonhost"
        if work == pid and NHP: work = NHP[0]   # the pool's non-host server (the process whose kernels are k_probe / k_first)
    # 2. a phase range of the same thread, else of the same process
    if lab is None:
        for same_tid in (True, False):
            for (p2, s, e, t, t2) in RT5:
                if p2 != pid or not t.startswith("phase: ") or (same_tid and t2 != htid): continue
                o = overlap(h0, h1, s, e)
                if o > 0: best[t[7:]] = best.get(t[7:], 0) + o
            if best: lab = max(best, key=best.get); break
    if lab is None: lab = "unlabeled"
    if lab == "prefill": lab = "map2" if "relock" in txt else "map1"
    if lab == "solo" and False: pass
    rows.append(dict(pid=pid, h0=h0, h1=h1, lab=lab, work=work, txt=txt))
labs = collections.Counter(r["lab"] for r in rows)
print("hold labels", dict(labs))

# map hold2 of a prefill can also be labelled solo: "(relock/Solo)" holds with phase solo stay solo.
# ---------------- per-label summary
def host_cover_range(pid, t):
    """innermost (shortest) non-lock NVTX range of the process covering t"""
    best = None; bd = 1 << 62
    for (p, s, e, tx) in R:
        if p != pid or s > t or e < t or tx.startswith("device lock"): continue
        if e - s < bd: bd = e - s; best = tx
    return best or "(no range)"
Rp = collections.defaultdict(list)
for r in R: Rp[r[0]].append(r)

def host_cover(pid, t):
    best = None; bd = 1 << 62
    for (p, s, e, tx) in Rp[pid]:
        if s > t or e < t or tx.startswith("device lock"): continue
        if e - s < bd: bd = e - s; best = tx
    return best or "(no range)"

RT = arr("select globalTid>>24,start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME", 4)
RTp = {p: RT[RT[:, 0] == p] for p in np.unique(RT[:, 0])} if len(RT) else {}
RTp = {p: v[np.argsort(v[:, 1])] for p, v in RTp.items()}
agg = collections.defaultdict(lambda: dict(n=0, hold=0, busy=0, kbusy=0, mbusy=0, idle=0, nocall=0, gapbins=[0, 0, 0, 0], cover=collections.Counter(), apigap=collections.Counter(), start=[]))
for r in rows:
    a = agg[r["lab"]]; w = r["work"]; h = (r["h1"] - r["h0"]) / 1e9
    ca = cov_all.get(w); ck = cov_k.get(w); cm = cov_m.get(w)
    busy = ca.cov(r["h0"], r["h1"]) / 1e9 if ca else 0
    a["n"] += 1; a["hold"] += h; a["busy"] += busy
    a["kbusy"] += (ck.cov(r["h0"], r["h1"]) / 1e9) if ck else 0
    a["mbusy"] += (cm.cov(r["h0"], r["h1"]) / 1e9) if cm else 0
    a["idle"] += h - busy
    gaps = ca.gaps(r["h0"], r["h1"]) if ca else [(r["h0"], r["h1"])]
    vv = RTp.get(r["work"]); selh = vv[(vv[:, 1] <= r["h1"]) & (vv[:, 2] >= r["h0"])] if vv is not None else np.zeros((0, 4), dtype=np.int64)
    for (g0, g1) in gaps:
        d = (g1 - g0) / 1e9
        a["gapbins"][0 if d < 1e-3 else 1 if d < 1e-2 else 2 if d < 1e-1 else 3] += d
        if d >= 0.005: a["cover"][host_cover(r["pid"], (g0 + g1) // 2)] += d
        if d >= 0.001 and len(selh):
            sg = selh[(selh[:, 1] < g1) & (selh[:, 2] > g0)]
            sg = sg[(sg[:, 2] - sg[:, 1]) > 50000]   # calls longer than 50 us
            if len(sg):
                cg = Cover(np.maximum(sg[:, 1], g0), np.minimum(sg[:, 2], g1)); a["nocall"] += d - cg.cov(g0, g1) / 1e9
                for x in sg: a["apigap"][S.get(int(x[3]), "?")] += (min(x[2], g1) - max(x[1], g0)) / 1e9
            else: a["nocall"] += d
        elif d >= 0.001: a["nocall"] += d
print("\n== slot holds by label (sums over the run; per hold = / n)")
print("%-12s %4s %8s %8s | %7s %7s %7s | %8s %8s | idle by gap size: <1ms %6s 1-10ms %6s 10-100ms %6s >100ms %6s" % ("label", "n", "hold s", "mean s", "busy%", "kern%", "copy%", "idle s", "idle/hold", "", "", "", ""))
tot_hold = sum(a["hold"] for a in agg.values())
for lab, a in sorted(agg.items(), key=lambda kv: -kv[1]["hold"]):
    n = a["n"]; hs = a["hold"]
    print("%-12s %4d %8.1f %8.2f | %6.1f%% %6.1f%% %6.1f%% | %8.1f %8.2f | %6.1f %6.1f %6.1f %6.1f" % (lab, n, hs, hs / n, 100 * a["busy"] / hs, 100 * a["kbusy"] / hs, 100 * a["mbusy"] / hs, a["idle"], a["idle"] / n, *[x for x in a["gapbins"]]))
print("total hold s %.1f" % tot_hold)
print("\n== what the host does while the slot is held and the GPU idle (gaps >= 5 ms, seconds over the run; innermost NVTX range of the holder)")
for lab, a in sorted(agg.items(), key=lambda kv: -kv[1]["hold"]):
    print("-- %s (n=%d)" % (lab, a["n"]))
    for t, d in a["cover"].most_common(8): print("     %8.1f s  %s" % (d, t))
    print("   idle gaps >= 1 ms: seconds with no CUDA call (> 50 us) by any thread of the process: %.1f s; CUDA calls overlapping the idle gaps:" % a["nocall"])
    for t, d in a["apigap"].most_common(6): print("     %8.1f s  %s" % (d, t))

# ---------------- kernels by label
def nm(i): return S.get(int(i), "?")
kn = collections.defaultdict(lambda: [0, 0])
kl = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
kpid = {p: K[K[:, 0] == p] for p in np.unique(K[:, 0])}
kst = {p: v[np.argsort(v[:, 1])] for p, v in kpid.items()}
intime = np.zeros(len(K), dtype=bool)
tot_by_lab = collections.Counter()
for r in rows:
    v = kst.get(r["work"])
    if v is None: continue
    lo = np.searchsorted(v[:, 1], r["h0"], side="left"); hi = np.searchsorted(v[:, 1], r["h1"], side="right")
    for rowk in v[lo:hi]:
        d = rowk[2] - rowk[1]; e = kl[r["lab"]][nm(rowk[4])]; e[0] += d; e[1] += 1; tot_by_lab[r["lab"]] += d
for row in K:
    e = kn[nm(row[4])]; e[0] += row[2] - row[1]; e[1] += 1
print("\n== GPU kernel time by hold label (s, sum over kernels; kernels of different streams overlap)")
for lab, d in tot_by_lab.most_common(): print("  %-10s %8.1f s" % (lab, d / 1e9))
print("\n== all kernels, all processes, top 40 by total time: name total_s count mean_us")
allk = sum(v[0] for v in kn.values())
for k, (d, n) in sorted(kn.items(), key=lambda kv: -kv[1][0])[:40]:
    print("  %7.2f s %5.1f%% %9d %9.1f us  %s" % (d / 1e9, 100 * d / allk, n, d / n / 1e3, k[:110]))
print("  total kernel time %.1f s" % (allk / 1e9))
for lab in kl:
    print("-- kernels in %s holds (top 8)" % lab)
    for k, (d, n) in sorted(kl[lab].items(), key=lambda kv: -kv[1][0])[:8]:
        print("     %7.2f s %9d x %8.1f us  %s" % (d / 1e9, n, d / n / 1e3, k[:100]))

# ---------------- copies by label
print("\n== copies inside holds by label: kind (1 H2D 2 D2H 8 D2D) pinned? bytes GB, time s (sum), GB/s")
cl = collections.defaultdict(lambda: [0, 0, 0])
mp = {p: M[M[:, 0] == p] for p in np.unique(M[:, 0])} if len(M) else {}
for r in rows:
    v = mp.get(r["work"])
    if v is None: continue
    sel = v[(v[:, 1] >= r["h0"]) & (v[:, 1] <= r["h1"])]
    for ck in np.unique(sel[:, 4]):
        for pin in (0, 1):
            ss = sel[(sel[:, 4] == ck) & ((sel[:, 6] == 1) == bool(pin) if ck == 1 else (sel[:, 7] == 1) == bool(pin))]
            if len(ss):
                e = cl[(r["lab"], int(ck), pin)]; e[0] += int(ss[:, 5].sum()); e[1] += int((ss[:, 2] - ss[:, 1]).sum()); e[2] += len(ss)
for (lab, ck, pin), (b, t, n) in sorted(cl.items()):
    print("  %-10s kind %d %s %8.2f GB %8.2f s %7.1f GB/s  %d copies" % (lab, ck, "pinned  " if pin else "pageable", b / 1e9, t / 1e9, b / max(t, 1), n))

# ---------------- runtime API per label
RT = arr("select globalTid>>24,start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME", 4)
def cls(n):
    if n.startswith("cudaMalloc") or n.startswith("cudaHostAlloc") or n.startswith("cuMemAlloc") or n.startswith("cudaMemPool") or n.startswith("cuMemCreate"): return "alloc"
    if n.startswith("cudaFree") or n.startswith("cuMemFree") : return "free"
    if "Synchronize" in n or n.startswith("cudaStreamWaitEvent"): return "sync"
    if n.startswith("cudaMemcpy") and "Async" not in n: return "memcpy(sync)"
    if n.startswith("cudaMemcpy"): return "memcpyAsync"
    if n.startswith("cudaMemset"): return "memset"
    if "Launch" in n: return "launch"
    return None
ncls = {i: cls(v) for i, v in S.items()}
api = collections.defaultdict(lambda: [0, 0])
RTp = {p: RT[RT[:, 0] == p] for p in np.unique(RT[:, 0])} if len(RT) else {}
byname = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for r in rows:
    v = RTp.get(r["pid"])
    if v is None: continue
    sel = v[(v[:, 1] >= r["h0"]) & (v[:, 1] <= r["h1"])]
    for x in sel:
        k = ncls.get(int(x[3]))
        if k is None: continue
        e = api[(r["lab"], k)]; e[0] += 1; e[1] += x[2] - x[1]
        if k in ("alloc", "free", "sync", "memcpy(sync)"): b = byname[(r["lab"], k)][S[int(x[3])]]; b[0] += 1; b[1] += x[2] - x[1]
print("\n== CUDA runtime calls inside holds, by label and class: calls, host seconds (the calling thread's wall in the call)")
for (lab, k), (n, t) in sorted(api.items()):
    print("  %-10s %-14s %9d calls %9.2f s   (per hold: %.1f calls)" % (lab, k, n, t / 1e9, n / agg[lab]["n"]))
print("\n-- alloc/free/sync names")
for (lab, k), d in sorted(byname.items()):
    print("  %s %s: %s" % (lab, k, ", ".join("%s x%d %.2fs" % (nme, n, t / 1e9) for nme, (n, t) in sorted(d.items(), key=lambda kv: -kv[1][1])[:5])))

# ---------------- GPU metrics
NAMES = {2: "GR active", 3: "SMs active", 4: "SM issue", 12: "warps in flight (occupancy)", 15: "unallocated warps in active SMs", 18: "DRAM read", 19: "DRAM write", 20: "PCIe RX (H2D)", 21: "PCIe TX (D2H)"}
types = sorted(r[0] for r in q("select distinct typeId from GPU_METRICS"))
print("\n== GPU metrics: devices (typeIds)", types)
span0 = min(r["h0"] for r in rows); span1 = max(r["h1"] for r in rows)
w0 = span0 + 0.15 * (span1 - span0); w1 = span1 - 0.15 * (span1 - span0)   # steady window: the middle 70 % of the profiled pool's holds
MS = {}
for ti, ty in enumerate(types):
    for mid in NAMES:
        a = np.array(q("select timestamp,value from GPU_METRICS where typeId=%d and metricId=%d order by timestamp" % (ty, mid)), dtype=np.float64).reshape(-1, 2)
        MS[(ti, mid)] = (a[:, 0], np.concatenate(([0], np.cumsum(a[:, 1]))))
def wmean(ti, mid, a, b):
    t, cs = MS[(ti, mid)]
    i = np.searchsorted(t, a); j = np.searchsorted(t, b)
    return (cs[j] - cs[i]) / (j - i) if j > i else float("nan")
print("steady window %.0f s (middle 70%% of pool-0 holds); mean over samples:" % ((w1 - w0) / 1e9))
print("  %-34s %8s %8s" % ("metric (% of peak unless noted)", "GPU0", "GPU1"))
for mid, nme in NAMES.items():
    print("  %-34s %8.1f %8.1f" % (nme, wmean(0, mid, w0, w1), wmean(1, mid, w0, w1) if len(types) > 1 else float("nan")))
# per hold label on GPU 0, only stretches where exactly one hold of the profiled pool is active
ev = []
for r in rows: ev.append((r["h0"], 1, r["lab"])); ev.append((r["h1"], -1, r["lab"]))
ev.sort(); active = collections.Counter(); segs = []; last = None
for (t, d, lab) in ev:
    if last is not None and t > last and sum(active.values()) == 1: segs.append((last, t, next(k for k, v in active.items() if v > 0)))
    active[lab] += d; last = t
print("\n== GPU0 metrics per hold label, only while exactly one pool-0 hold is active (n segments, seconds, then means)")
print("  %-10s %5s %7s | %s" % ("label", "segs", "sec", " ".join("%9s" % NAMES[m][:9] for m in NAMES)))
by = collections.defaultdict(list)
for s in segs: by[s[2]].append(s)
for lab, ss in sorted(by.items(), key=lambda kv: -sum(x[1] - x[0] for x in kv[1])):
    sec = sum(x[1] - x[0] for x in ss) / 1e9; vals = []
    for mid in NAMES:
        tot = 0; cnt = 0
        for (a, b, _) in ss:
            t, cs = MS[(0, mid)]; i = np.searchsorted(t, a); j = np.searchsorted(t, b)
            tot += cs[j] - cs[i]; cnt += j - i
        vals.append(tot / cnt if cnt else float("nan"))
    print("  %-10s %5d %7.1f | %s" % (lab, len(ss), sec, " ".join("%9.1f" % v for v in vals)))

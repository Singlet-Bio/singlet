#!/usr/bin/env python3
"""E21 (docs/gpu-engine/fable/E21-pool.md): E18's sampler (bench/fable/throughput/e18_sampler.py) for `singlet gpu-pool`, one JSON line
per tick, in E18's format (e18_point.py, e18_steady.py and e18_locks.py read it unchanged).

Usage: e21_sampler.py OUT.jsonl CGROUP_DIR [INTERVAL_S=0.5] [PSS_EVERY_S=4] [GPU_EVERY_S=2] [REPORT_DIR ...]

A process belongs to a copy (a sample's run) by
  - a pool's REPORT_DIR/pids.tsv (pid, label, start time; the pool writes one line per process it forks: pool<D>, nhsrv<D>, c<j>) with
    REPORT_DIR/gmap.tsv (j -> the job's item number i: label c<i>); a process not listed takes the label of its nearest listed
    ancestor (ppid chain: the pipeline's STAR child, the CR4 probes, sort, ...). The pool's own process and its non-host server are
    "copies" of their own (pool<D>, nhsrv<D>);
  - else, as E18, E18_COPY=<copy> in the process's environment (the run script's single runs).
Per tick: t, cpu, mem, mstat (anon / file / shmem / anon_thp), copies {cpu, rss, n}, locks {h, w} as E18, tick_s (the tick's own
time). Memory and the GPU are read by two threads so that the tick stays at INTERVAL_S with many processes (E18's sampler stretched
past 5 s with 32 runs): every PSS_EVERY_S the copies' smaps_rollup (pss; E21's priv = Private_Clean + Private_Dirty and panon =
Pss_Anon; GB; pss_s = the sweep's time) and every GPU_EVERY_S nvidia-smi (gpu, gproc), each attached to the first tick after it ends.
"""
import sys, os, time, json, glob, subprocess, threading

out, cg = sys.argv[1], sys.argv[2]
dt = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
pss_every = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
gpu_every = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
reports = sys.argv[6:]
HZ = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")
uid = os.getuid()


def rd(p):
    try:
        with open(p, "rb") as f:
            return f.read()
    except OSError:
        return None


def cg_cpu():
    b = rd(f"{cg}/cpu.stat")
    if b:
        for l in b.decode().splitlines():
            if l.startswith("usage_usec"):
                return int(l.split()[1]) / 1e6
    return None


def cg_mem():
    cur = rd(f"{cg}/memory.current")
    st = rd(f"{cg}/memory.stat")
    d = {}
    if st:
        for l in st.decode().splitlines():
            k, v = l.split()[:2]
            if k in ("anon", "file", "shmem", "anon_thp"):
                d[k] = round(int(v) / 2**30, 2)
    return (round(int(cur) / 2**30, 2) if cur else None), d


# ---- the pools' processes: (pid, start time) -> label
listed = {}
offsets = {}
gmaps = {}


def read_reports():
    for r in reports:
        if not gmaps.get(r):
            g = {}
            b = rd(f"{r}/gmap.tsv")
            if b:
                for l in b.decode().splitlines():
                    f = l.split("\t")
                    if len(f) >= 2:
                        g[f[0]] = f[1]
            gmaps[r] = g
        b = rd(f"{r}/pids.tsv")
        if not b:
            continue
        o = offsets.get(r, 0)
        if len(b) <= o:
            continue
        chunk = b[o:]
        last = chunk.rfind(b"\n")
        if last < 0:
            continue
        offsets[r] = o + last + 1
        for l in chunk[:last].decode(errors="replace").splitlines():
            f = l.split("\t")
            if len(f) < 3 or not f[0].isdigit():
                continue
            lab = f[1]
            if lab.startswith("c") and lab[1:].isdigit():
                lab = "c" + gmaps[r].get(lab[1:], lab[1:] + "?")
            listed[(int(f[0]), f[2])] = lab


env_copy = {}   # (pid, start) -> E18_COPY value or None


def env_label(pid, start):
    k = (pid, start)
    if k in env_copy:
        return env_copy[k]
    c = None
    env = rd(f"/proc/{pid}/environ")
    if env:
        for kv in env.split(b"\0"):
            if kv.startswith(b"E18_COPY="):
                c = kv[9:].decode()
                break
    env_copy[k] = c
    return c


label_cache = {}   # (pid, start) -> label (positive results only: a pool's pids.tsv line may come a tick late)


def scan():
    """this tick's processes of the user: pid -> (ppid, start, stat fields, statm); and pid -> label (None: not a copy's)"""
    procs = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        pid = int(d)
        try:
            if os.stat(f"/proc/{d}").st_uid != uid:
                continue
        except OSError:
            continue
        st = rd(f"/proc/{d}/stat")
        sm = rd(f"/proc/{d}/statm")
        if not st or not sm:
            continue
        f = st[st.rfind(b")") + 2:].split()
        # fields after "(comm) " (0-based): state 0, ppid 1, ..., utime 11, stime 12, cutime 13, cstime 14, ..., starttime 19
        procs[pid] = (int(f[1]), f[19].decode(), f, sm)
    lab = {}

    def label(pid, depth=0):
        if pid in lab:
            return lab[pid]
        p = procs.get(pid)
        if p is None or depth > 64:
            return None
        k = (pid, p[1])
        c = label_cache.get(k) or listed.get(k)
        if c is None:
            c = env_label(pid, p[1])
        if c is None:
            c = label(p[0], depth + 1)
        if c is not None:
            label_cache[k] = c
        lab[pid] = c
        return c

    for pid in procs:
        label(pid)
    alive = set((pid, p[1]) for pid, p in procs.items())
    for k in list(label_cache):
        if k not in alive:
            del label_cache[k]
    for k in list(env_copy):
        if k not in alive:
            del env_copy[k]
    return procs, lab


# ---- the memory thread: smaps_rollup of every labelled process, summed per copy
cur_labels = {}   # pid -> label, the main loop's latest
lock = threading.Lock()
mem_new = None


def mem_thread():
    global mem_new
    while True:
        t0 = time.time()
        with lock:
            pids = dict(cur_labels)
        m = {}
        for pid, c in pids.items():
            sr = rd(f"/proc/{pid}/smaps_rollup")
            if not sr:
                continue
            e = m.setdefault(c, {"pss": 0.0, "priv": 0.0, "panon": 0.0})
            for l in sr.decode(errors="replace").splitlines():
                if l.startswith("Pss:"):
                    e["pss"] += int(l.split()[1]) / 2**20
                elif l.startswith("Pss_Anon:"):
                    e["panon"] += int(l.split()[1]) / 2**20
                elif l.startswith("Private_Clean:") or l.startswith("Private_Dirty:"):
                    e["priv"] += int(l.split()[1]) / 2**20
        for e in m.values():
            for k in e:
                e[k] = round(e[k], 2)
        with lock:
            mem_new = (round(time.time() - t0, 2), m)
        time.sleep(max(0.2, pss_every - (time.time() - t0)))


gpu_new = None


def gpu_thread():
    global gpu_new
    while True:
        t0 = time.time()
        g, gp = None, {}
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            g, bus = [], {}
            for l in r.stdout.strip().splitlines():
                i, b, u, mm = [x.strip() for x in l.split(",")]
                g.append([int(u), int(mm)])
                bus[b] = int(i)
            r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            with lock:
                labs = dict(cur_labels)
            for l in r.stdout.strip().splitlines():
                p = [x.strip() for x in l.split(",")]
                if len(p) != 3 or not p[1].isdigit():
                    continue
                c = labs.get(int(p[1])) or ("pid" + p[1])
                d = gp.setdefault(c, {})
                k = str(bus.get(p[0], p[0]))
                d[k] = d.get(k, 0) + (int(p[2]) if p[2].isdigit() else 0)
        except Exception as e:   # the sampler must never stop the run
            gp["err"] = str(e)[:80]
        with lock:
            gpu_new = (g, gp)
        time.sleep(max(0.1, gpu_every - (time.time() - t0)))


threading.Thread(target=mem_thread, daemon=True).start()
threading.Thread(target=gpu_thread, daemon=True).start()

with open(out, "a", buffering=1) as fo:
    while True:
        t0 = time.time()
        read_reports()
        rec = {"t": round(t0, 3), "cpu": cg_cpu()}
        rec["mem"], rec["mstat"] = cg_mem()
        procs, lab = scan()
        cp = {}
        for pid, (ppid, start, f, sm) in procs.items():
            c = lab.get(pid)
            if c is None:
                continue
            e = cp.setdefault(c, {"cpu": 0.0, "rss": 0.0, "n": 0})
            e["cpu"] += (int(f[11]) + int(f[12]) + int(f[13]) + int(f[14])) / HZ
            e["rss"] += int(sm.split()[1]) * PAGE / 2**30
            e["n"] += 1
        with lock:
            cur_labels.clear()
            cur_labels.update({pid: c for pid, c in lab.items() if c is not None})
            mn, mem_new = mem_new, None
            gn, gpu_new = gpu_new, None
        for e in cp.values():
            e["cpu"] = round(e["cpu"], 2)
            e["rss"] = round(e["rss"], 2)
        if mn:
            rec["pss_s"] = mn[0]
            for c, m in mn[1].items():
                cp.setdefault(c, {"cpu": 0.0, "rss": 0.0, "n": 0}).update(m)
        rec["copies"] = cp
        # engine device locks (/proc/locks: holder and waiters per lock file)
        ino = {}
        for p in glob.glob("/tmp/singlet_engine_gpu_*.lock"):
            try:
                s = os.stat(p)
            except OSError:
                continue
            ino["%02x:%02x:%d" % (os.major(s.st_dev), os.minor(s.st_dev), s.st_ino)] = os.path.basename(p)[len("singlet_engine_gpu_"):-len(".lock")]
        lk = {}
        b = rd("/proc/locks")
        if b and ino:
            for l in b.decode(errors="replace").splitlines():
                tk = l.split()
                blocked = len(tk) > 1 and tk[1] == "->"
                if blocked:
                    tk = tk[:1] + tk[2:]
                if len(tk) < 6 or tk[1] != "FLOCK":
                    continue
                name = ino.get(tk[5])
                if name is None:
                    continue
                c = lab.get(int(tk[4])) or ("pid" + tk[4])
                e = lk.setdefault(name, {"h": None, "w": []})
                if blocked:
                    e["w"].append(c)
                else:
                    e["h"] = c
        rec["locks"] = lk
        if gn:
            rec["gpu"], rec["gproc"] = gn
        rec["tick_s"] = round(time.time() - t0, 3)
        fo.write(json.dumps(rec, separators=(",", ":")) + "\n")
        time.sleep(max(0.0, dt - (time.time() - t0)))

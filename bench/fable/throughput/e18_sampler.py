#!/usr/bin/env python3
"""E18 (docs/gpu-engine/fable/E18-throughput.md): a sampler for k samples in flight on one node, one JSON line per tick.

Usage: e18_sampler.py OUT.jsonl CGROUP_DIR [INTERVAL_S=0.5] [GPU_EVERY=2]

Per tick (INTERVAL_S):
  t        wall clock (s)
  cpu      the job's cgroup CPU time (cpu.stat usage_usec, s)
  mem      the job's cgroup memory.current (GB); anon / file / shmem from memory.stat (GB)
  copies   per sample copy (processes whose environment has E18_COPY=<copy>, i.e. the run script's subshell, time, timeout, the
           pipeline, STAR and its children): cpu = sum of utime + stime + cutime + cstime of the live processes (a reaped child's
           time moves to its reaper's cutime, so the sum does not double count), rss = sum of VmRSS (GB; forked processes share
           pages, so this over-counts), pss (GB, every 4th tick: smaps_rollup), n = processes
  locks    per engine lock file /tmp/singlet_engine_gpu_<pci>[.k].lock (singlet_engine_gpu::lockDevice / lockDeviceSlot): the copy that
           holds it (h) and the copies blocked on it (w), from /proc/locks
  gpu      every GPU_EVERY ticks: per device [util %, memory used MB]; gproc: per copy its device memory (MB) by device index
           (nvidia-smi --query-compute-apps)
The run script starts it before a point and kills it after.
"""
import sys, os, time, json, glob, subprocess

out, cg = sys.argv[1], sys.argv[2]
dt = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
gevery = int(sys.argv[4]) if len(sys.argv) > 4 else 2
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
            if k in ("anon", "file", "shmem"):
                d[k] = round(int(v) / 2**30, 2)
    return (round(int(cur) / 2**30, 2) if cur else None), d


copy_of = {}   # pid -> copy (cached: the environment of a process does not change)


def copy_for(pid):
    if pid in copy_of:
        return copy_of[pid]
    c = None
    try:
        if os.stat(f"/proc/{pid}").st_uid == uid:
            env = rd(f"/proc/{pid}/environ")
            if env:
                for kv in env.split(b"\0"):
                    if kv.startswith(b"E18_COPY="):
                        c = kv[9:].decode()
                        break
    except OSError:
        return None
    copy_of[pid] = c
    return c


def gpu_query():
    g, gp = None, {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        g, bus = [], {}
        for l in r.stdout.strip().splitlines():
            i, b, u, m = [x.strip() for x in l.split(",")]
            g.append([int(u), int(m)])
            bus[b] = int(i)
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        for l in r.stdout.strip().splitlines():
            p = [x.strip() for x in l.split(",")]
            if len(p) != 3 or not p[1].isdigit():
                continue
            c = copy_for(int(p[1])) or ("pid" + p[1])
            d = gp.setdefault(c, {})
            k = str(bus.get(p[0], p[0]))
            d[k] = d.get(k, 0) + int(p[2]) if p[2].isdigit() else d.get(k, 0)
    except Exception as e:   # the sampler must never stop the run
        gp["err"] = str(e)[:80]
    return g, gp


tick = 0
with open(out, "a", buffering=1) as fo:
    while True:
        t0 = time.time()
        rec = {"t": round(t0, 3), "cpu": cg_cpu()}
        rec["mem"], rec["mstat"] = cg_mem()
        # processes of the copies
        cp = {}
        alive = set()
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            pid = int(d)
            c = copy_for(pid)
            if c is None:
                continue
            st = rd(f"/proc/{pid}/stat")
            sm = rd(f"/proc/{pid}/statm")
            if not st or not sm:
                continue
            alive.add(pid)
            f = st[st.rfind(b")") + 2:].split()
            # fields after "(comm) ": state=0 ... utime=11 stime=12 cutime=13 cstime=14 (0-based here)
            cpu = (int(f[11]) + int(f[12]) + int(f[13]) + int(f[14])) / HZ
            rss = int(sm.split()[1]) * PAGE / 2**30
            e = cp.setdefault(c, {"cpu": 0.0, "rss": 0.0, "n": 0})
            e["cpu"] += cpu
            e["rss"] += rss
            e["n"] += 1
            if tick % 4 == 0:
                sr = rd(f"/proc/{pid}/smaps_rollup")
                if sr:
                    for l in sr.decode(errors="replace").splitlines():
                        if l.startswith("Pss:"):
                            e["pss"] = e.get("pss", 0.0) + int(l.split()[1]) / 2**20
                            break
        for k in list(copy_of):
            if k not in alive and not os.path.exists(f"/proc/{k}"):
                del copy_of[k]
        for e in cp.values():
            e["cpu"] = round(e["cpu"], 2)
            e["rss"] = round(e["rss"], 2)
            if "pss" in e:
                e["pss"] = round(e["pss"], 2)
        rec["copies"] = cp
        # engine device locks
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
                c = copy_for(int(tk[4])) or ("pid" + tk[4])
                e = lk.setdefault(name, {"h": None, "w": []})
                if blocked:
                    e["w"].append(c)
                else:
                    e["h"] = c
        rec["locks"] = lk
        if tick % gevery == 0:
            rec["gpu"], rec["gproc"] = gpu_query()
        fo.write(json.dumps(rec, separators=(",", ":")) + "\n")
        tick += 1
        time.sleep(max(0.0, dt - (time.time() - t0)))

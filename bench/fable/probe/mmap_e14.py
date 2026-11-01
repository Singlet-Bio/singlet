#!/usr/bin/env python3
"""E14 (docs/gpu-engine/fable/E14-probe.md, section 5): the mapped alternatives to a private genome copy, on one node.
1. mmap(MAP_SHARED | MAP_POPULATE) of the index files straight from the page cache: time, and what page sizes the mapping gets;
2. the SA copied to a /dev/shm file and mapped (shmem pages: what a SysV segment or a tmpfs file gives);
3. a private copy on transparent hugepages (numpy madvises large arrays; what star_genome_preload makes).
For each, random 8-byte reads from the SA (one thread, the same 20 M positions): the page-size cost that STAR's seed search pays.
Usage: mmap_e14.py GENOME_DIR [N_RANDOM]. The /dev/shm file is unlinked as soon as it is mapped."""
import mmap, os, sys, time
import numpy as np

d = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000_000
POP = getattr(mmap, "MAP_POPULATE", 0x8000)


def smaps():
    keep = ("Rss", "AnonHugePages", "FilePmdMapped", "ShmemPmdMapped", "Shared_Clean", "Private_Clean", "Shmem")
    r = {}
    for line in open("/proc/self/smaps_rollup"):
        k, _, v = line.partition(":")
        if k.strip() in keep:
            r[k.strip()] = int(v.split()[0]) // 1024
    return r


def gather(v, idx, what):
    out = []
    for _ in range(3):
        t = time.time()
        s = int(v[idx].sum())
        out.append(time.time() - t)
    print(f"  random 8-byte reads, {what}: {len(idx) / 1e6:.0f} M in {min(out):.2f} s (best of 3; {min(out) / len(idx) * 1e9:.1f} ns each)")
    return s


maps = {}
t0 = time.time()
for f in ("Genome", "SA", "SAindex"):
    fd = os.open(f"{d}/{f}", os.O_RDONLY)
    t = time.time()
    maps[f] = mmap.mmap(fd, 0, flags=mmap.MAP_SHARED | POP, prot=mmap.PROT_READ)
    os.close(fd)
    print(f"1. mmap+populate {f} from the page cache: {len(maps[f]) / 1e9:.2f} GB in {time.time() - t:.2f} s")
print(f"   all three {time.time() - t0:.2f} s; smaps_rollup MB {smaps()}")
sa = np.frombuffer(maps["SA"], dtype=np.uint8)
m = (len(sa) // 8) * 8
v = sa[:m].view(np.uint64)
idx = np.random.default_rng(1).integers(0, len(v), n)
s1 = gather(v, idx, "file mapping (page cache)")

t = time.time()
p = "/dev/shm/e14-mmap-%d" % os.getpid()
fd = os.open(p, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
try:
    os.ftruncate(fd, m)
    sm = mmap.mmap(fd, m, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
finally:
    os.close(fd)
    os.unlink(p)   # the mapping keeps the pages until this process exits
w = np.frombuffer(sm, dtype=np.uint64)
w[:] = v
print(f"2. SA copied into a /dev/shm file and mapped: {time.time() - t:.2f} s; smaps_rollup MB {smaps()}")
s2 = gather(w, idx, "shmem mapping")
del w
sm.close()

t = time.time()
cp = np.empty(len(v), dtype=np.uint64)
cp[:] = v
print(f"3. private copy: {time.time() - t:.2f} s; smaps_rollup MB {smaps()}")
s3 = gather(cp, idx, "private copy")
assert s1 == s2 == s3

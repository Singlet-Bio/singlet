#!/usr/bin/env python3
"""t2-cachetest-fincore.py [--drop] [--tsv FILE] SPEC...   (throughput track 2, cachetest)
Page-cache residency of files, per category, without util-linux fincore: mmap(PROT_READ, MAP_SHARED) + mincore() through ctypes.
SPEC = name:depth:path  (path is a file, or a directory walked to `depth` levels of subdirectories; depth 0 = the files directly in it).
A file is counted once, in the first category that reaches it (by device+inode). Symbolic links are followed for files only.
Output: one line per category "CAT name files=N cached_gb=X size_gb=Y", per file with cached pages (--tsv): name, cached bytes, size, path;
final line "TOTAL cached_gb=X". --drop: after counting, posix_fadvise(DONTNEED) on every file of the categories (clean pages only, no root).
Cap: MAXFILES (default 6000) files per category."""
import ctypes, mmap, os, sys, stat

libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
PAGE = mmap.PAGESIZE
MAXFILES = int(os.environ.get("MAXFILES", "6000"))


def resident(path):
    """(cached bytes, size) of a regular file"""
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            return 0, 0
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr is None or addr == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "mmap")
        try:
            n = (size + PAGE - 1) // PAGE
            vec = ctypes.create_string_buffer(n)
            if libc.mincore(addr, size, vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore")
            pages = vec.raw.count(bytes([1]))   # Linux sets bit 0 only
        finally:
            libc.munmap(addr, size)
        return min(pages * PAGE, size), size
    finally:
        os.close(fd)


def walk(path, depth):
    if os.path.isfile(path):
        yield path
        return
    stack = [(path, 0)]
    while stack:
        d, lv = stack.pop()
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for nm in names:
            p = os.path.join(d, nm)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if stat.S_ISDIR(st.st_mode):
                if lv < depth:
                    stack.append((p, lv + 1))
            elif stat.S_ISREG(st.st_mode) or (stat.S_ISLNK(st.st_mode) and os.path.isfile(p)):
                yield p


def main():
    args = sys.argv[1:]
    drop = False
    tsv = None
    while args and args[0].startswith("--"):
        a = args.pop(0)
        if a == "--drop":
            drop = True
        elif a == "--tsv":
            tsv = open(args.pop(0), "a")
    seen = set()
    tot = 0
    for spec in args:
        name, depth, path = spec.split(":", 2)
        nf = cb = sb = 0
        for p in walk(path, int(depth)):
            if nf >= MAXFILES:
                print("# %s: capped at %d files" % (name, MAXFILES))
                break
            try:
                st = os.stat(p)
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                c, s = resident(p)
                if drop:
                    fd = os.open(p, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
            except OSError as e:
                print("# %s: %s: %s" % (name, p, e))
                continue
            nf += 1
            cb += c
            sb += s
            if tsv and c > 0:
                tsv.write("%s\t%d\t%d\t%s\n" % (name, c, s, p))
        tot += cb
        print("CAT %s files=%d cached_gb=%.2f size_gb=%.2f" % (name, nf, cb / 2**30, sb / 2**30))
    print("TOTAL cached_gb=%.2f" % (tot / 2**30))


main()

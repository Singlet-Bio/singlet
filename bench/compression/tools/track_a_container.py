#!/usr/bin/env python
"""Track A deliverable 6, part 2: which zstd build wrote each .1pz?

For every file, every zstd frame of the original (perm, ptr, chunk payloads, colsums, metadata) is decompressed and
re-compressed at level 3 with checksum on (what pz_writer.h does) by every zstd library installed on this machine.
A library "reproduces" a file when it regenerates every frame byte for byte.

    python -P tools/track_a_container.py [corpus/*/*_counts.1pz]
"""
import ctypes, glob, os, struct, sys
import zstandard

HDR = "<IHBBIIQBBHIIIIIQIIQII"

LIBS = {
    "zstd-1.5.5 (Rtools static, harness)": None,   # evaluated by harness/track_a_container.exe
    "python-zstandard 0.25 (bundled)": "py",
    "Git for Windows mingw64 libzstd.dll": r"C:\Program Files\Git\mingw64\bin\libzstd.dll",
    "Tesseract-OCR libzstd.dll": r"C:\Program Files\Tesseract-OCR\libzstd.dll",
}


class CLib:
    def __init__(self, path):
        self.l = ctypes.CDLL(path)
        l = self.l
        l.ZSTD_versionString.restype = ctypes.c_char_p
        l.ZSTD_createCCtx.restype = ctypes.c_void_p
        l.ZSTD_CCtx_setParameter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        l.ZSTD_CCtx_setParameter.restype = ctypes.c_size_t
        l.ZSTD_CCtx_reset.argtypes = [ctypes.c_void_p, ctypes.c_int]
        l.ZSTD_compress2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        l.ZSTD_compress2.restype = ctypes.c_size_t
        l.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
        l.ZSTD_compressBound.restype = ctypes.c_size_t
        l.ZSTD_isError.argtypes = [ctypes.c_size_t]
        self.ctx = l.ZSTD_createCCtx()
        self.version = l.ZSTD_versionString().decode()

    def compress(self, data, level=3, checksum=1, **params):
        l = self.l
        l.ZSTD_CCtx_reset(self.ctx, 3)  # session and parameters
        l.ZSTD_CCtx_setParameter(self.ctx, 100, level)
        l.ZSTD_CCtx_setParameter(self.ctx, 201, checksum)
        for k, v in params.items():
            l.ZSTD_CCtx_setParameter(self.ctx, k, v)
        buf = ctypes.create_string_buffer(int(l.ZSTD_compressBound(len(data))))
        n = l.ZSTD_compress2(self.ctx, buf, len(buf), data, len(data))
        if l.ZSTD_isError(n):
            raise RuntimeError("zstd error")
        return buf.raw[:n]


class PyLib:
    def __init__(self):
        self.version = ".".join(map(str, zstandard.ZSTD_VERSION))

    def compress(self, data, level=3, checksum=1, **params):
        return zstandard.ZstdCompressor(level=level, write_checksum=bool(checksum), write_content_size=True).compress(data)


def frames(buf):
    (magic, ver, vt, flags, m, n, nnz, pw, lvl, _p, nch, perm_z, ptr_z, chunk_cols, ff, meta_off, meta_z, colsums_z, t_off, t_z,
     t_ch) = struct.unpack(HDR, buf[:80])
    out = []
    off = 96
    out.append(("perm", buf[off:off + perm_z])); off += perm_z
    out.append(("ptr", buf[off:off + ptr_z])); off += ptr_z
    ct = struct.unpack("<%dI" % nch, buf[off:off + 4 * nch]); off += 4 * nch
    for c in range(nch):
        out.append(("chunk", buf[off + 20:off + ct[c]])); off += ct[c]
    out.append(("colsums", buf[off:off + colsums_z])); off += colsums_z
    out.append(("meta", buf[off:off + meta_z]))
    return out


def bundle_main(path):
    import zipfile
    z = zipfile.ZipFile(path)
    libs = {"1.5.7": PyLib(), "1.5.6": CLib(LIBS["Git for Windows mingw64 libzstd.dll"])}
    dec = zstandard.ZstdDecompressor()
    print("bundle %s: frames reproduced at level 3 by zstd 1.5.6 (== 1.5.5 on these inputs) / 1.5.7" % path)
    for i in z.infolist():
        if not i.filename.endswith(".1pz"):
            continue
        raw = z.read(i)
        try:
            fr = frames(raw)
        except Exception:
            continue
        if len(fr) <= 2 or len(raw) < 1000:
            continue
        r = {}
        for v, lib in libs.items():
            ok = 0
            for kind, zz in fr:
                ok += lib.compress(dec.decompress(zz, max_output_size=1 << 31)) == zz
            r[v] = "%d/%d" % (ok, len(fr))
        print("%-52s 1.5.6: %-8s 1.5.7: %s" % (i.filename, r["1.5.6"], r["1.5.7"]), flush=True)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--bundle":
        return bundle_main(sys.argv[2])
    files = []
    for a in (sys.argv[1:] or ["corpus/*/exon_counts.1pz", "corpus/*/intron_counts.1pz"]):
        files += glob.glob(a)
    libs = {}
    for name, path in LIBS.items():
        if path is None:
            continue
        try:
            libs[name] = PyLib() if path == "py" else CLib(path)
        except Exception as e:
            print("cannot load", name, e)
    for name, lib in libs.items():
        print("lib %-40s version %s" % (name, lib.version))
    dec = zstandard.ZstdDecompressor()
    print("\nper file: fraction of original frames reproduced byte-for-byte at level 3 + checksum (frames > 128 KiB shown separately)")
    hdr = "%-34s" % "file" + "".join("%-34s" % ("%s %s" % (n.split()[0][:14], l.version)) for n, l in libs.items())
    print(hdr)
    summary = {}
    for f in sorted(files):
        raw = open(f, "rb").read()
        fr = frames(raw)
        res = []
        for name, lib in libs.items():
            ok = big_ok = big = 0
            for kind, z in fr:
                data = dec.decompress(z, max_output_size=1 << 31)
                same = lib.compress(data) == z
                ok += same
                if len(data) > 131072:
                    big += 1
                    big_ok += same
            res.append("%d/%d (big %d/%d)" % (ok, len(fr), big_ok, big))
        key = os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f)
        print("%-34s" % key + "".join("%-34s" % r for r in res))


if __name__ == "__main__":
    main()

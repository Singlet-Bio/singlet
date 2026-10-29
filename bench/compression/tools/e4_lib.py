"""Shared helpers for the E4 tools (derivability audit, tiered layout, latency model).

Everything read from corpus/ is untrusted input: it is only ever parsed (struct, zstandard with an output cap,
numpy); nothing from it is executed. Run with `python -P`.

Layout of a .1pz file (writer order, include/singlet/pileup/pz_writer.h):
    header(96) | perm_z | ptr_z | chunk table (4 B per chunk) | chunk blobs | colsums_z | metadata_z | footer(16)
"""
import hashlib
import importlib.util
import os
import struct
import subprocess
import sys
import zipfile

import numpy as np
import zstandard

HERE = os.path.dirname(os.path.abspath(__file__))
BC = os.path.dirname(HERE)  # bench/compression
ROOT = os.path.dirname(os.path.dirname(BC))  # repo root
RESULTS = os.path.join(BC, "results")
BUNDLE = os.path.join(BC, "corpus", "bundles", "GSE145370.singlet")
SCRATCH = (
    r"C:\Users\zacha\AppData\Local\Temp\claude\C--Users-zacha-Projects-singlet-bio"
    r"\65f6415a-aab4-4a48-a77c-a89863f1e1ea\scratchpad\e4"
)

CORPUS = [
    "pbmc_v2_typical",
    "pbmc_v2_shallow",
    "pbmc_v3_lowgenes",
    "brain_sn_v3_deep",
    "brain_sn_v3_intronic",
    "tcell_5p",
    "tumor_3p",
]

MAX_OUT = 1 << 30  # decompression output cap for untrusted frames


def load_pz():
    """Load the compiled reader directly (avoids importing the whole singlet package, which takes minutes here)."""
    pyd = os.path.join(ROOT, "python", "singlet", "_pz.cp311-win_amd64.pyd")
    spec = importlib.util.spec_from_file_location("_pz", pyd)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HDR_FMT = "<IHBBIIQBBHIIIIIQIIQII16s"
HDR_FIELDS = (
    "magic version vt_code flags m n nnz ptr_width codec_level pad0 num_chunks perm_z_sz ptr_z_sz chunk_cols "
    "feature_flags metadata_offset metadata_z_sz colsums_z_sz transpose_offset transpose_z_sz transpose_chunks reserved"
).split()
assert struct.calcsize(HDR_FMT) == 96

FLAG_HAS_PERM, FLAG_GAP16, FLAG_HAS_METADATA, FLAG_HAS_COLSUMS = 0x01, 0x02, 0x04, 0x10


def zdec(buf, off, size):
    """Decompress one zstd frame buf[off:off+size] with an output cap."""
    return zstandard.ZstdDecompressor().decompress(bytes(buf[off : off + size]), max_output_size=MAX_OUT)


def parse_pz(buf):
    """Locate every block of a .1pz image held in memory. Returns a dict of header fields and block offsets."""
    if len(buf) < 96 + 16:
        raise ValueError("short .1pz")
    h = dict(zip(HDR_FIELDS, struct.unpack_from(HDR_FMT, buf, 0)))
    if h["magic"] != 0x5A315054:
        raise ValueError("bad magic")
    nc = h["num_chunks"]
    o = {}
    o["perm_off"] = 96
    o["ptr_off"] = 96 + h["perm_z_sz"]
    o["ctable_off"] = o["ptr_off"] + h["ptr_z_sz"]
    ctab = np.frombuffer(buf, dtype="<u4", count=nc, offset=o["ctable_off"]).astype(np.int64)
    o["ctable"] = ctab
    o["chunks_off"] = o["ctable_off"] + 4 * nc
    o["chunks_bytes"] = int(ctab.sum())
    o["colsums_off"] = o["chunks_off"] + o["chunks_bytes"]
    o["meta_off"] = o["colsums_off"] + h["colsums_z_sz"]
    o["footer_off"] = len(buf) - 16
    o["n_bytes"] = len(buf)
    o["unaccounted"] = o["footer_off"] - (o["meta_off"] + h["metadata_z_sz"])
    return h, o


def parse_tlv(meta_raw):
    """Metadata TLV: tag(1) len(4) payload; tag 1 rownames, 2 colnames (NUL-terminated strings), 3 user kv."""
    out = {"rownames": [], "colnames": [], "kv": b""}
    p = 0
    n = len(meta_raw)
    while p < n:
        tag = meta_raw[p]
        p += 1
        if tag == 0:
            break
        (ln,) = struct.unpack_from("<I", meta_raw, p)
        p += 4
        payload = meta_raw[p : p + ln]
        p += ln
        if tag in (1, 2):
            names = payload.decode("utf-8", "replace").split("\0")
            if names and names[-1] == "":
                names.pop()
            out["rownames" if tag == 1 else "colnames"] = names
        elif tag == 3:
            out["kv"] = bytes(payload)
    return out


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def zip_inventory(path):
    """Per-entry layout of a zip64 archive: names, methods, sizes, header/data offsets. Also central-directory size."""
    z = zipfile.ZipFile(path)
    rows = []
    cd_bytes = 0
    with open(path, "rb") as f:
        for zi in z.infolist():
            f.seek(zi.header_offset)
            lh = f.read(30)
            sig, ver, flag, comp, mt, md, crc, csz, usz, nl, el = struct.unpack("<IHHHHHIIIHH", lh)
            assert sig == 0x04034B50
            rows.append(
                dict(
                    name=zi.filename,
                    method=zi.compress_type,
                    csize=zi.compress_size,
                    usize=zi.file_size,
                    hdr_off=zi.header_offset,
                    lh_len=30 + nl + el,
                    data_off=zi.header_offset + 30 + nl + el,
                    cd_rec=46 + len(zi.filename.encode()) + len(zi.extra) + len(zi.comment),
                )
            )
            cd_bytes += rows[-1]["cd_rec"]
    size = os.path.getsize(path)
    tail = size - z.start_dir  # central directory + zip64 EOCD record + locator + EOCD
    z.close()
    return rows, dict(file_size=size, start_dir=z.start_dir, cd_bytes=cd_bytes, tail_bytes=tail)


# --------------------------------------------------------------------------------------------------------------------
# Reference C++ helper: calls the *production* compute_psi() and write_1pz() so that the Python recomputation is
# compared with the production code itself, not with a re-implementation of it.
# --------------------------------------------------------------------------------------------------------------------
HELPER_SRC = r'''
// E4 reference helper (built by tools/e4_lib.py into the scratchpad; not part of the repo)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "singlet/pileup/pz_writer.h"
#include "singlet/pileup/splice_psi.h"
using namespace singlet;
struct CSC { std::vector<int32_t> indptr, indices; std::vector<uint16_t> data; uint32_t nrows, ncols; };
static std::vector<std::string> lines(const std::string& p) {
    std::ifstream f(p, std::ios::binary); std::string s((std::istreambuf_iterator<char>(f)), {});
    std::vector<std::string> v; size_t a = 0;
    while (a < s.size()) { size_t b = s.find('\n', a); if (b == std::string::npos) b = s.size(); v.push_back(s.substr(a, b - a)); a = b + 1; }
    return v;
}
int main(int argc, char** argv) {
    std::string mode = argv[1];
    if (mode == "zc3") {  // zc3 in out : zstd level 3 + checksum exactly as pz_writer.h does it
        std::ifstream f(argv[2], std::ios::binary); std::string s((std::istreambuf_iterator<char>(f)), {});
        pz::ZstdCCtx cctx;
        auto z = pz::zstd_compress(cctx.ctx, (const uint8_t*)s.data(), s.size(), 3);
        std::ofstream o(argv[3], std::ios::binary); o.write((const char*)z.data(), z.size());
        return 0;
    }
    // psi dir : read dir/sj.bin, dir/names.txt, dir/cols.txt ; write dir/psi.bin and dir/psi_ref.1pz
    std::string d = argv[2];
    CSC sj; uint64_t nnz;
    { std::ifstream f(d + "/sj.bin", std::ios::binary);
      f.read((char*)&sj.nrows, 4); f.read((char*)&sj.ncols, 4); f.read((char*)&nnz, 8);
      sj.indptr.resize(sj.ncols + 1); sj.indices.resize(nnz); sj.data.resize(nnz);
      f.read((char*)sj.indptr.data(), 4 * (sj.ncols + 1)); f.read((char*)sj.indices.data(), 4 * nnz);
      f.read((char*)sj.data.data(), 2 * nnz); }
    auto names = lines(d + "/names.txt"); auto cols = lines(d + "/cols.txt");
    PSIResult r = compute_psi(sj, names);
    { std::ofstream o(d + "/psi.bin", std::ios::binary);
      uint64_t k = r.data.size(); o.write((const char*)&k, 8);
      o.write((const char*)r.indptr.data(), 4 * r.indptr.size());
      o.write((const char*)r.indices.data(), 4 * r.indices.size());
      o.write((const char*)r.data.data(), 4 * r.data.size()); }
    bool ok = pz::write_1pz(d + "/psi_ref.1pz", r.nrows, r.ncols, r.indptr, r.indices, r.data, names, cols, 3, 1024, 1, {});
    // variant: values rounded to nearest-even before the integer cast (what the bundle's stored values look like)
    std::vector<uint32_t> rd(r.data.size());
    for (size_t i = 0; i < rd.size(); ++i) rd[i] = (uint32_t)std::nearbyint(r.data[i]);
    bool ok2 = pz::write_1pz(d + "/psi_ref_rint.1pz", r.nrows, r.ncols, r.indptr, r.indices, rd, names, cols, 3, 1024, 1, {});
    std::printf("events=%zu psi_nnz=%zu write_ok=%d,%d\n", r.events.size(), r.data.size(), (int)ok, (int)ok2);
    return (ok && ok2) ? 0 : 1;
}
'''


def build_helper():
    """Compile the reference helper with Rtools GCC (static zstd 1.5.5). Returns path to the exe."""
    os.makedirs(SCRATCH, exist_ok=True)
    src = os.path.join(SCRATCH, "e4_ref.cpp")
    exe = os.path.join(SCRATCH, "e4_ref.exe")
    old = open(src).read() if os.path.exists(src) else None
    if old != HELPER_SRC:
        with open(src, "w") as f:
            f.write(HELPER_SRC)
    if old == HELPER_SRC and os.path.exists(exe) and os.path.getmtime(exe) >= os.path.getmtime(src):
        return exe
    tc = r"C:\RBuildTools\4.4\x86_64-w64-mingw32.static.posix"
    env = dict(os.environ)
    env["PATH"] = tc + r"\bin;C:\RBuildTools\4.4\usr\bin;" + env["PATH"]
    cmd = [
        tc + r"\bin\g++.exe", "-std=c++20", "-O2", "-DNDEBUG", "-I", os.path.join(ROOT, "include"),
        "-I", tc + r"\include", src, "-o", exe, tc + r"\lib\libzstd.a", "-lpthread", "-static",
    ]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("helper build failed:\n" + r.stderr[-3000:])
    return exe


def zstd_version_of_helper(exe):
    return "1.5.5 (Rtools static libzstd)"

"""I/O helpers shared by the barcode-layer harness scripts: FASTQ -> reads.bin, 2-bit / 3-bit packing of string arrays,
whitelist loading, and STAR BAM tag extraction. numpy only (plus samtools for the BAM)."""
from __future__ import annotations

import struct
import subprocess
from itertools import islice
from pathlib import Path

import numpy as np

MAGIC = 0x31524C42  # 'BLR1'
_CODE2 = np.full(256, 4, dtype=np.uint8)
for _i, _c in enumerate("ACGT"):
    _CODE2[ord(_c)] = _i
    _CODE2[ord(_c.lower())] = _i
_CODE3 = np.full(256, 5, dtype=np.uint8)           # umi_key: A C G T N other
for _i, _c in enumerate("ACGT"):
    _CODE3[ord(_c)] = _i
    _CODE3[ord(_c.lower())] = _i
_CODE3[ord("N")] = 4
_CODE3[ord("n")] = 4


def fastq_to_reads(fq: str, out: str, qual_const: str | None = None, max_reads: int | None = None) -> int:
    """Writes reads.bin from a FASTQ (4 lines per record). qual_const replaces every quality with one character, as singlet
    does for the barcode read (R1 quality 'F'). Returns the number of reads."""
    seqs: list[np.ndarray] = []
    quals: list[np.ndarray] = []
    lens: list[np.ndarray] = []
    n = 0
    with open(fq, "rb") as f:
        while True:
            take = 4_000_000 if max_reads is None else min(4_000_000, 4 * (max_reads - n))
            if take <= 0:
                break
            lines = list(islice(f, take))
            if not lines:
                break
            s = lines[1::4]
            q = lines[3::4]
            ln = np.fromiter((len(x) - 1 for x in s), dtype=np.int64, count=len(s))
            sa = np.frombuffer(b"".join(s), dtype=np.uint8)
            sa = sa[sa != 10]
            if qual_const is None:
                qa = np.frombuffer(b"".join(q), dtype=np.uint8)
                qa = qa[qa != 10]
            else:
                qa = np.full(sa.shape, ord(qual_const), dtype=np.uint8)
            seqs.append(sa)
            quals.append(qa)
            lens.append(ln)
            n += len(s)
    ln = np.concatenate(lens) if lens else np.zeros(0, np.int64)
    off = np.zeros(len(ln) + 1, dtype=np.uint64)
    np.cumsum(ln, out=off[1:])
    with open(out, "wb") as o:
        o.write(struct.pack("<IIQ", MAGIC, 0, len(ln)))
        o.write(off.tobytes())
        for a in seqs:
            o.write(a.tobytes())
        for a in quals:
            o.write(a.tobytes())
    return len(ln)


def load_reads(path: str):
    """Returns (off uint64[n+1], seq uint8[], qual uint8[])."""
    with open(path, "rb") as f:
        magic, _, n = struct.unpack("<IIQ", f.read(16))
        assert magic == MAGIC
        off = np.frombuffer(f.read(8 * (n + 1)), dtype=np.uint64)
        total = int(off[-1])
        seq = np.frombuffer(f.read(total), dtype=np.uint8)
        qual = np.frombuffer(f.read(total), dtype=np.uint8)
    return off, seq, qual


def fixed_matrix(off, arr, start: int, length: int, pad: int = ord("N")):
    """(n, length) uint8 matrix of arr[off[i]+start : +length]; positions past the read end are `pad` (STAR pads with N / 'H')."""
    n = len(off) - 1
    ln = (off[1:] - off[:-1]).astype(np.int64)
    out = np.full((n, length), pad, dtype=np.uint8)
    for j in range(length):
        ok = ln > start + j
        out[ok, j] = arr[(off[:-1].astype(np.int64) + start + j)[ok]]
    return out


def pack2_matrix(m: np.ndarray):
    """2-bit value (N as A) and N position per row, STAR convertNuclStrToInt64: pos -1 none, >=0 one N, -2 several."""
    c = _CODE2[m]
    isn = c > 3
    nn = isn.sum(axis=1)
    pos = np.where(nn == 0, -1, np.where(nn == 1, isn.argmax(axis=1), -2)).astype(np.int64)
    v = np.zeros(len(m), dtype=np.uint64)
    cc = np.where(isn, 0, c).astype(np.uint64)
    for j in range(m.shape[1]):
        v = (v << np.uint64(2)) + cc[:, j]
    return v, pos


def pack3_matrix(m: np.ndarray):
    v = np.zeros(len(m), dtype=np.uint64)
    c = _CODE3[m].astype(np.uint64)
    for j in range(m.shape[1]):
        v = (v << np.uint64(3)) | c[:, j]
    return v


def strings_to_matrix(a: np.ndarray, length: int):
    """array of bytes strings (dtype S) -> (n, length) uint8, short strings padded with 0."""
    a = a.astype(f"S{length}")
    return np.frombuffer(a.tobytes(), dtype=np.uint8).reshape(len(a), length)


def decode2(v: np.ndarray, length: int) -> np.ndarray:
    """2-bit values -> bytes strings (dtype S<length>)."""
    tab = np.frombuffer(b"ACGT", dtype=np.uint8)
    m = np.empty((len(v), length), dtype=np.uint8)
    x = v.copy()
    for j in range(length - 1, -1, -1):
        m[:, j] = tab[(x & np.uint64(3)).astype(np.int64)]
        x >>= np.uint64(2)
    return np.frombuffer(m.tobytes(), dtype=f"S{length}")


def decode3(v: np.ndarray, length: int) -> np.ndarray:
    tab = np.frombuffer(b"ACGTN?", dtype=np.uint8)
    m = np.empty((len(v), length), dtype=np.uint8)
    x = v.copy()
    for j in range(length - 1, -1, -1):
        m[:, j] = tab[np.minimum(x & np.uint64(7), np.uint64(5)).astype(np.int64)]
        x >>= np.uint64(3)
    return np.frombuffer(m.tobytes(), dtype=f"S{length}")


def read_lines(path: str) -> list[str]:
    out = []
    with open(path) as f:
        for s in f:
            s = s.strip()
            if s:
                out.append(s)
    return out


def wl_keys(path: str):
    """Whitelist file -> (sorted unique 2-bit keys uint64, original strings for the kept entries sorted by key). Entries with N
    at position 0 are skipped, any other N reads as A (ParametersSolo.cpp)."""
    lines = read_lines(path)
    L = len(lines[0])
    assert all(len(x) == L for x in lines), "mixed lengths: use the per-length segment path"
    m = np.frombuffer("".join(lines).encode(), dtype=np.uint8).reshape(len(lines), L)
    keep = _CODE2[m[:, 0]] <= 3
    v, _ = pack2_matrix(m[keep])
    keys = np.unique(v)
    return keys, L


def write_u64_list(path: str, keys: np.ndarray):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(keys)))
        f.write(np.ascontiguousarray(keys, dtype=np.uint64).tobytes())


def load_arr(prefix: str, name: str, typ: str):
    dt = {"i64": "<i8", "i32": "<i4", "u64": "<u8", "u32": "<u4"}[typ]
    return np.fromfile(f"{prefix}.{name}.{typ}", dtype=dt)


def star_tags(bam: str, tags=("CB", "UB", "UR", "CR")):
    """Primary records of a STAR BAM -> dict of arrays keyed by read index (read names are r<i>). Missing tags are b''."""
    if bam.endswith(".sam"):           # local self-test: header-less SAM text
        class _P:
            stdout = (l for l in open(bam, "rb") if not l.startswith(b"@"))
            def wait(self): pass
        p = _P()
    else:
        p = subprocess.Popen(["samtools", "view", "-F", "0x900", bam], stdout=subprocess.PIPE, bufsize=1 << 22)
    idx, flag = [], []
    cols = {t: [] for t in tags}
    want = {t.encode() + b":Z:": t for t in tags}
    for line in p.stdout:
        f = line.rstrip(b"\n").split(b"\t")
        idx.append(int(f[0][1:]))
        flag.append(int(f[1]))
        got = {}
        for x in f[11:]:
            t = want.get(x[:5])
            if t is not None:
                got[t] = x[5:]
        for t in tags:
            cols[t].append(got.get(t, b""))
    p.wait()
    out = {"idx": np.array(idx, dtype=np.int64), "flag": np.array(flag, dtype=np.int32)}
    for t in tags:
        out[t] = np.array(cols[t], dtype="S64")
    return out


def write_reads_from_strings(path: str, seqs: list[str]):
    ln = np.array([len(s) for s in seqs], dtype=np.uint64)
    off = np.zeros(len(seqs) + 1, dtype=np.uint64)
    np.cumsum(ln, out=off[1:])
    blob = "".join(seqs).encode()
    with open(path, "wb") as o:
        o.write(struct.pack("<IIQ", MAGIC, 0, len(seqs)))
        o.write(off.tobytes())
        o.write(blob)
        o.write(b"F" * len(blob))


def read_fastq_seqs(path: str, n: int) -> list[str]:
    """First n sequences of a (gzipped) FASTQ."""
    import gzip
    op = gzip.open if path.endswith(".gz") else open
    out = []
    with op(path, "rb") as f:
        for i, line in enumerate(f):
            if i % 4 == 1:
                out.append(line.strip().decode())
                if len(out) >= n:
                    break
    return out

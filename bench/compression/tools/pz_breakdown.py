#!/usr/bin/env python
"""Where do the bytes of a .1pz go? Header, permutation, column pointers, metadata (names),
column sums, and the per-chunk VOCSC blobs. Reads only the 96-byte header plus the file size.

    python -P tools/pz_breakdown.py corpus/*/*_counts.1pz
"""
import glob, os, struct, sys

HDR = "<IHBBIIQBBHIIIIIQIIQII"  # 80 bytes of fields + 16 reserved


def breakdown(path):
    sz = os.path.getsize(path)
    with open(path, "rb") as f:
        hdr = f.read(96)
    (magic, ver, vt, flags, m, n, nnz, pw, lvl, _pad, nch, perm_z, ptr_z, chunk_cols, ff,
     meta_off, meta_z, colsums_z, t_off, t_z, t_ch) = struct.unpack(HDR, hdr[:80])
    assert magic == 0x5A315054, path
    # layout: header, perm, meta?, colsums?, chunk table (u32 per chunk), chunks, footer(16)
    chunks_z = sz - 96 - perm_z - ptr_z - meta_z - colsums_z - 4 * nch - 16
    return dict(file=path, bytes=sz, m=m, n=n, nnz=nnz, vt=vt, level=lvl, chunk_cols=chunk_cols,
                chunks=nch, perm_z=perm_z, ptr_z=ptr_z, meta_z=meta_z, colsums_z=colsums_z,
                chunks_z=chunks_z)


def main():
    paths = []
    for a in sys.argv[1:]:
        paths += glob.glob(a)
    cols = ["sample", "member", "bytes", "m", "n", "nnz", "chunks", "perm_z", "ptr_z", "meta_z",
            "colsums_z", "chunks_z", "meta_pct", "bits_per_nnz", "chunk_bits_per_nnz"]
    print("\t".join(cols))
    for p in sorted(paths):
        b = breakdown(p)
        sample = os.path.basename(os.path.dirname(p))
        member = os.path.basename(p)
        row = [sample, member, b["bytes"], b["m"], b["n"], b["nnz"], b["chunks"], b["perm_z"], b["ptr_z"],
               b["meta_z"], b["colsums_z"], b["chunks_z"], f"{b['meta_z'] / b['bytes']:.1%}",
               f"{b['bytes'] * 8 / b['nnz']:.2f}", f"{b['chunks_z'] * 8 / b['nnz']:.2f}"]
        print("\t".join(str(x) for x in row))


if __name__ == "__main__":
    main()

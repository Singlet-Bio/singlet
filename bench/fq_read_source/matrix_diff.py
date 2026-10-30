"""Value-level difference of the count matrices of two Singlet runs (out/ directories, --output-format mtx).

    python matrix_diff.py RUN_A/out RUN_B/out

For every <name>.mtx.gz in both directories whose bytes differ: columns are aligned by barcode (<name>_barcodes.tsv.gz),
rows by feature name (<name>_features.tsv.gz). Prints totals, the summed absolute difference relative to A's total,
the fraction of A's non-zero entries that changed, and the number of barcodes only in A / only in B. Counts only;
nothing per cell or per feature is printed.
"""
from __future__ import annotations

import gzip
import hashlib
import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def names(p: Path) -> list[str]:
    if not p.exists():
        return []
    with gzip.open(p, "rt") as f:
        return [line.rstrip("\n").split("\t")[0] for line in f]


def load(d: Path, stem: str):
    m = sp.csc_matrix(scipy.io.mmread(str(d / f"{stem}.mtx.gz")))
    return m, names(d / f"{stem}_features.tsv.gz"), names(d / f"{stem}_barcodes.tsv.gz")


def main(a: Path, b: Path) -> None:
    stems = sorted({p.name[:-7] for p in a.glob("*.mtx.gz")} & {p.name[:-7] for p in b.glob("*.mtx.gz")})
    print("matrix diffs (A vs B): total A, total B, |A-B|/total A, changed/nnz A, barcodes only A / only B")
    for s in stems:
        if sha(a / f"{s}.mtx.gz") == sha(b / f"{s}.mtx.gz"):
            continue
        try:
            ma, fa, ba = load(a, s)
            mb, fb, bb = load(b, s)
        except Exception as e:  # noqa: BLE001
            print(f"  {s}: unreadable ({e})")
            continue
        fa = fa or [str(i) for i in range(ma.shape[0])]
        fb = fb or [str(i) for i in range(mb.shape[0])]
        ba = ba or [str(i) for i in range(ma.shape[1])]
        bb = bb or [str(i) for i in range(mb.shape[1])]
        rows = sorted(set(fa) | set(fb))
        cols = sorted(set(ba) | set(bb))
        ri = {n: i for i, n in enumerate(rows)}
        ci = {n: i for i, n in enumerate(cols)}

        def embed(m, f, bcs):
            m = m.tocoo()
            r = np.array([ri[f[i]] for i in m.row], dtype=np.int64)
            c = np.array([ci[bcs[j]] for j in m.col], dtype=np.int64)
            return sp.csr_matrix((m.data.astype(np.float64), (r, c)), shape=(len(rows), len(cols)))

        A, B = embed(ma, fa, ba), embed(mb, fb, bb)
        D = (A - B).tocsr()
        D.eliminate_zeros()
        ta, tb = A.sum(), B.sum()
        print(f"  {s}: {ta:.0f} {tb:.0f} {abs(D).sum() / max(ta, 1):.3e} {D.nnz / max(A.nnz, 1):.3e} "
              f"{len(set(ba) - set(bb))} / {len(set(bb) - set(ba))}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

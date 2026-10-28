""".s1 sample objects written with the singlet package's own codec and container code.

There is no public writer yet (PLAN.md gap G2), so this builds the object the way the package does internally:
`singlet.s1.codec.NativeCodec` (codec "auto": lane rANS, or the generic codec if that is smaller) for the pz2 member and
`singlet.s1.container.S2Writer` for the sample object. When `singlet.write_s1` exists, call it here instead and check
that the sizes do not change.

`s1` stores gene ids and barcodes inline. `s1_pack` stores the gene ids as a reference to a shared vocabulary pack (a
file every sample of the same annotation shares and a reader downloads once); the pack is written next to the object
and reported as shared bytes, not as part of the object's size.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..util import Canon
from . import GROUP_S1, Format

MEMBER = "counts"


def _csc(c: Canon):
    from singlet.s1.matrix import VT_U32, Csc

    return Csc(
        m=c.m,
        n=c.n,
        indptr=c.indptr.astype(np.int64),
        indices=c.indices.astype(np.uint32),
        data=c.data.astype(np.uint32),
        rownames=[str(g) for g in c.genes],
        colnames=[str(b) for b in c.barcodes],
        value_type=VT_U32,
    )


def pack_id(genes) -> bytes:
    return hashlib.sha256(("\0".join(map(str, genes)) + "\0").encode()).digest()[:8]


def encode_member(c: Canon, threads: int, row_names="inline") -> tuple[bytes, bytes]:
    from singlet.s1.codec import EncodeOptions, NativeCodec
    from singlet.s1.matrix import canonical_digest

    csc = _csc(c)
    digest = canonical_digest(csc)
    body = NativeCodec("auto").encode(csc, EncodeOptions(row_names=row_names, col_names="inline", threads=threads, digest=digest))
    return body, digest


class S1(Format):
    id, label, ext, group = "s1", ".s1", ".s1", GROUP_S1
    use_pack = False

    def write(self, c, path, threads):
        from singlet.s1.container import S2Writer

        if self.use_pack:
            pid = pack_id(c.genes)
            body, digest = encode_member(c, threads, row_names=("pack", pid))
            self._pack_path(path).write_text("\n".join(map(str, c.genes)) + "\n")
        else:
            body, digest = encode_member(c, threads)
        w = S2Writer("bench")
        w.add_matrix(MEMBER, body, digest, {"modality": "counts"})
        path.write_bytes(w.build())

    def _pack_path(self, path: Path) -> Path:
        return path.with_suffix(".genes.pack.txt")

    def shared_bytes(self, path):
        p = self._pack_path(path)
        return p.stat().st_size if self.use_pack and p.exists() else 0

    def read(self, path, threads, mode: str = "canonical", isa=None) -> Canon:
        """Whole-matrix decode through the compiled codec; `mode='canonical'` restores the source row order."""
        from singlet import _pz2  # type: ignore
        from singlet.s1.container import S2Reader

        rd = S2Reader(str(path))
        body = rd.read_member(MEMBER)
        mem = _pz2.Member(body)
        out = mem.decode(mode=mode, threads=threads, isa=isa)
        rows = mem.row_names()
        cols = mem.col_names()
        genes = rows.get("names") if isinstance(rows, dict) else rows
        if genes is None and self.use_pack:
            genes = self._pack_path(path).read_text().splitlines()
        barcodes = cols.get("names") if isinstance(cols, dict) else cols
        return Canon(
            indptr=np.asarray(out["indptr"], dtype=np.int64),
            indices=np.asarray(out["indices"], dtype=np.uint32),
            data=np.asarray(out["data"], dtype=np.uint32),
            genes=np.asarray(genes),
            barcodes=np.asarray(barcodes),
        )

    def decode_only(self, body: bytes, threads: int, mode: str, isa=None) -> int:
        """Decode a member already in memory; returns nnz. This is what the codec-speed numbers time."""
        from singlet import _pz2  # type: ignore

        out = _pz2.decode(body, mode=mode, threads=threads, isa=isa)
        return int(out["nnz"])


class S1Pack(S1):
    id, label = "s1_pack", ".s1 (shared gene pack)"
    use_pack = True


FORMATS: list[Format] = [S1(), S1Pack()]

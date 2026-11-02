""".s1 sample objects written by the singlet package's public writer, `singlet.write_s1` (codec "auto": lane rANS, or the
generic codec if that is smaller), with one matrix member, `counts`.

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


def pack_id(genes) -> bytes:
    return hashlib.sha256(("\0".join(map(str, genes)) + "\0").encode()).digest()[:8]


class S1(Format):
    id, label, ext, group = "s1", ".s1", ".s1", GROUP_S1
    use_pack = False

    def write(self, c, path, threads):
        import scipy.sparse as sp
        from singlet import write_s1

        x = sp.csc_matrix((c.data, c.indices, c.indptr), shape=(c.m, c.n)).T  # cells x genes, as write_s1 takes it
        pid = None
        if self.use_pack:
            pid = pack_id(c.genes)
            self._pack_path(path).write_text("\n".join(map(str, c.genes)) + "\n")
        write_s1(x, path, genes=[str(g) for g in c.genes], barcodes=[str(b) for b in c.barcodes], sample_id="bench",
                 member=MEMBER, gene_pack=pid, threads=threads)

    def _pack_path(self, path: Path) -> Path:
        return path.with_suffix(".genes.pack.txt")

    def shared_bytes(self, path):
        p = self._pack_path(path)
        return p.stat().st_size if self.use_pack and p.exists() else 0

    def load(self, path, threads, mode: str = "canonical", isa=None):
        """Whole-matrix decode through the compiled codec into CSC arrays with gene ids and barcodes, which is what
        `decode` times; `mode='canonical'` restores the source row order (the order every other format returns)."""
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
        return out, genes, barcodes

    def to_canon(self, obj) -> Canon:
        out, genes, barcodes = obj
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

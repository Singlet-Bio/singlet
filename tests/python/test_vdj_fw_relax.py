"""Terminal anchor of the V(D)J assembly graph: a mutated Phe118 codon is accepted only with fw_relax."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from singlet.vdj import _kernels as K  # noqa: E402

_C = {"A": 0, "C": 1, "G": 2, "T": 3}


def _codes(s):
    return np.array([_C[c] for c in s], np.uint8)


def test_canonical_codon_unchanged():
    s = _codes("ACCTGGGTGTTCGGCGGAGGGACC")  # ...TGG GTG TTC GGC GGA GGG: Phe at 9
    assert K._codon_at(s, 0, len(s), 9, 1, False) == 9
    assert K._codon_at(s, 0, len(s), 9, 1, True) == 9


def test_mutated_phe_needs_relax():
    s = _codes("ACCTGGGTGGTCGGCGGAGGGACC")  # Phe118 TTC -> GTC (Val)
    assert K._codon_at(s, 0, len(s), 9, 1, False) == -1
    assert K._codon_at(s, 0, len(s), 9, 1, True) == 9


def test_relax_rejects_stop_and_broken_motif():
    stop = _codes("ACCTGGGTGTAGGGCGGAGGGACC")  # TAG
    assert K._codon_at(stop, 0, len(stop), 9, 1, True) == -1
    nomotif = _codes("ACCTGGGTGGTCAACGGAGGGACC")  # no Gly after the codon
    assert K._codon_at(nomotif, 0, len(nomotif), 9, 1, True) == -1
    far = _codes("ACCTGGGTGGTCGGCAAAAAAACC")  # Gly-X-Gly broken at +9
    assert K._codon_at(far, 0, len(far), 9, 1, True) == -1


def test_cys_unaffected_by_relax():
    s = _codes("GGGTGTAAAA")
    assert K._codon_at(s, 0, len(s), 3, 0, True) == 3
    mut = _codes("GGGTCTAAAA")
    assert K._codon_at(mut, 0, len(mut), 3, 0, True) == -1


def test_mutated_phe_chain_is_assembled_and_productive():
    """A somatic Phe118 -> Val substitution (TTT -> GTT) in every transcript of one J gene: with asm_fw_relax and
    fw_mut_productive (the defaults) the chain is assembled, ends at the mutated codon and is productive; without them it is lost."""
    import dataclasses

    import pytest

    pytest.importorskip("numba")
    from singlet.adt import seq as S
    from singlet.vdj import assemble as A

    import test_vdj_assemble as T

    segs = T.make_segments()
    mutated = {"TRAJ1", "TRBJ1"}
    sim = []
    for s in segs:
        if s.name in mutated:
            assert s.seq[s.anchor : s.anchor + 3] == "TTT"
            s = dataclasses.replace(s, seq=s.seq[: s.anchor] + "GTT" + s.seq[s.anchor + 3 :])
        sim.append(s)
    r1, r2, truth, cbs = T.simulate(sim)
    ref = A.Reference(segs)  # germline: Phe
    wl, _ = S.encode_strings(cbs, 16)
    lacking = {k: t for k, t in truth.items() if t[2] in mutated}
    assert lacking and all(t[0].endswith("GTT") for t in lacking.values())

    def run(**kw):
        res = A.run_library([], [], ref=ref, cb_len=16, umi_len=10, whitelist_codes=np.sort(wl), chunks=[(r1, r2)],
                            params=A.Params(threads=2, **kw), log=lambda *a: None)
        k = res.contigs[res.contigs.kept & res.contigs.productive]
        return {(b[:-2], c): n for b, c, n in k[["barcode", "chain", "cdr3_nt"]].itertuples(index=False)}

    on = run(asm_fw_relax=True, fw_mut_productive=True)
    off = run(asm_fw_relax=False, fw_mut_productive=False)
    assert sum(on.get(k) == t[0] for k, t in lacking.items()) >= 0.9 * len(lacking)
    assert sum(off.get(k) == t[0] for k, t in lacking.items()) == 0
    # chains with a canonical Phe/Trp118 are unchanged
    other = {k: t for k, t in truth.items() if t[2] not in mutated}
    assert sum(on.get(k) == t[0] for k, t in other.items()) >= sum(off.get(k) == t[0] for k, t in other.items())

"""Reference fragment / peak reducers (bench/atac_reducers): equality with the CPU ATAC semantics, invariance under shuffled
input, byte-identical output for K = 1, 2, 7 cell partitions including the canonical merge of the peak set."""
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench" / "atac_reducers"))
import atac_ref as R  # noqa: E402
import barcode_ref as B  # noqa: E402
import sim  # noqa: E402

KS = (1, 2, 7)


@pytest.fixture(scope="module")
def data():
    rec, planted = sim.make_records()
    return rec, planted


def eq(a: dict, b: dict, keys=None):
    for k in keys or a:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=k)


def test_fixture_is_small(data):
    assert sim.nbytes(data[0]) < 1_000_000


def test_poisson_tail_matches_long_double():
    """The C++ test's reference: forward sum of the pmf in long double."""
    def ref(k, lam):
        lam = np.longdouble(lam)
        term = np.exp(-lam)
        for i in range(1, k + 1):
            term *= lam / i
        s = np.longdouble(0)
        for i in range(k, k + 2000):
            s += term
            term *= lam / (i + 1)
        return float(-np.log10(s))
    for k, lam in [(10, 2.0), (50, 5.0), (8, 0.3), (30, 12.0), (3, 0.5)]:
        got, want = R.poisson_upper_neg_log10(k, lam), ref(k, lam)
        assert abs(got - want) <= 1e-6 * abs(want)
    assert R.poisson_upper_neg_log10(0, 3.0) == 0.0
    assert R.poisson_upper_neg_log10(5, 0.0) == 300.0


# ---- fragments ---------------------------------------------------------------------------------------------------
def test_fragments_equal_cpu_on_coordinate_sorted_input(data):
    rec, _ = data
    cpu = R.cpu_fragments(sim.coord_sorted(rec))                      # what the CPU sees: a sorted BAM
    red = R.coord_order(R.fragment_reduce(R.fragment_emit(rec)))      # reducer: any order, written in the CPU's file order
    eq(cpu, red, ("tid", "start", "end", "cell", "read_pairs"))
    assert len(cpu["cell"]) > 1000 and cpu["read_pairs"].max() > 1


def test_fragments_invariant_under_shuffle(data):
    rec, _ = data
    base = R.fragment_reduce(R.fragment_emit(rec))
    for s in range(3):
        eq(base, R.fragment_reduce(R.fragment_emit(sim.shuffled(rec, s))))


def test_cpu_dedup_is_order_dependent_when_unsorted(data):
    """Finding F1: the CPU clears its dedup table when (contig, start) changes, so on an unsorted stream duplicates are kept."""
    rec, _ = data
    sorted_n = len(R.cpu_fragments(sim.coord_sorted(rec))["cell"])
    shuf_n = len(R.cpu_fragments(sim.shuffled(rec, 1))["cell"])
    assert shuf_n > sorted_n
    assert len(R.fragment_reduce(R.fragment_emit(sim.shuffled(rec, 1)))["cell"]) == sorted_n


def test_fragment_geometry_is_leftmost_mate_for_either_read1(data):
    rec, _ = data
    t = R.fragment_emit(rec)
    # every emitted fragment is a left mate's [pos+4, pos+tlen-5): the left mate is READ1 in about half of the pairs
    left = (rec["tlen"] > 0)
    assert abs(((rec["flag"] & R.F_READ1) != 0)[left].mean() - 0.5) < 0.1
    assert len(t["cell"]) > 0 and np.all(t["end"] > t["start"])


@pytest.mark.parametrize("K", KS)
def test_fragments_partition_invariant(data, K):
    rec, _ = data
    one = R.fragment_reduce(R.fragment_emit(rec))
    parts = [R.fragment_reduce(R.fragment_emit(R.select(rec, R.cell_part(rec["cell"], K) == p))) for p in range(K)]
    assert sum(len(p["cell"]) for p in parts) == len(one["cell"])
    assert R.to_bytes(R.fragment_merge(parts)) == R.to_bytes(one)
    cells = [set(p["cell"].tolist()) for p in parts]
    assert sum(len(c) for c in cells) == len(set().union(*cells))      # partitions are disjoint in cell


# ---- peaks -------------------------------------------------------------------------------------------------------
def _frags_and_table(rec):
    f = R.fragment_reduce(R.fragment_emit(rec))
    return f, R.insertion_reduce(R.insertion_emit(f, sim.IS_MITO))


def test_peaks_equal_cpu_port_and_find_planted(data):
    rec, planted = data
    f, table = _frags_and_table(rec)
    red = R.peak_call(table, sim.CHROM_LEN, sim.IS_MITO)
    # CPU port gets the insertions the CPU builds: start and end - 1 of every unique non-mito fragment, any order
    ins = [[] for _ in sim.CHROM_LEN]
    for t, s, e in zip(f["tid"], f["start"], f["end"]):
        if not sim.IS_MITO[t]:
            ins[t] += [int(s), int(e) - 1]
    cpu = R.cpu_peak_call(ins, [int(x) for x in sim.CHROM_LEN], list(sim.IS_MITO))
    eq(cpu, red)
    n = len(red["tid"])
    assert n >= 8
    hit = sum(any(red["tid"][i] == t and red["start"][i] - 200 <= c <= red["end"][i] + 200 for i in range(n)) for t, c in planted)
    assert hit >= 0.8 * len(planted)
    assert np.all(red["start"][1:][red["tid"][1:] == red["tid"][:-1]] > red["end"][:-1][red["tid"][1:] == red["tid"][:-1]])   # disjoint


def test_insertion_table_merge_is_canonical(data):
    rec, _ = data
    f, one = _frags_and_table(rec)
    for K in KS:
        parts = []
        for p in range(K):
            fp = R.fragment_reduce(R.fragment_emit(R.select(rec, R.cell_part(rec["cell"], K) == p)))
            parts.append(R.insertion_reduce(R.insertion_emit(fp, sim.IS_MITO)))
        assert R.to_bytes(R.insertion_merge(parts)) == R.to_bytes(one)


def test_peak_counts_and_metrics_equal_cpu(data):
    rec, _ = data
    out = R.run_all(rec, 1, sim.CHROM_LEN, sim.IS_MITO, sim.NCELLS)
    f = R.coord_order(out["fragments"])
    cpu_m, cpu_mat = R.cpu_cell_metrics_and_matrix(f, out["peaks"], sim.IS_MITO, sim.NCELLS)
    eq(cpu_m, out["metrics"])
    eq(cpu_mat, out["peak_counts"])
    assert out["peak_counts"]["count"].sum() > 500
    assert out["metrics"]["peak_region_fragments"].sum() >= out["metrics"]["peak_region_cutsites"].sum() / 2


# ---- the whole thing: K and order ---------------------------------------------------------------------------------
def digest(out):
    return hashlib.sha256(R.to_bytes(out)).hexdigest()


def test_end_to_end_byte_identical_over_K_and_order(data):
    rec, _ = data
    ref = digest(R.run_all(rec, 1, sim.CHROM_LEN, sim.IS_MITO, sim.NCELLS))
    for K in KS:
        for s in (None, 11, 12):
            r = rec if s is None else sim.shuffled(rec, s)
            assert digest(R.run_all(r, K, sim.CHROM_LEN, sim.IS_MITO, sim.NCELLS)) == ref, (K, s)


def test_unbalanced_partitions_and_empty_partition(data):
    """K larger than the cell count leaves empty partitions; the merge must not care."""
    rec, _ = data
    small = R.select(rec, rec["cell"] < 3)
    ref = digest(R.run_all(small, 1, sim.CHROM_LEN, sim.IS_MITO, sim.NCELLS))
    assert digest(R.run_all(small, 7, sim.CHROM_LEN, sim.IS_MITO, sim.NCELLS)) == ref


# ---- A1 (CPU port of the new default) ------------------------------------------------------------------------------------
def test_a1_cpu_global_dedup_port_is_order_free_and_equals_sorted_cpu(data):
    rec, _ = data
    sorted_old = R.cpu_fragments(sim.coord_sorted(rec))                 # PR #48 on a sorted BAM
    for r in (rec, sim.shuffled(rec, 3), sim.coord_sorted(rec)):
        new = R.cpu_fragments(r, assume_sorted=False)                  # branch claude/atac-orderfree-a1-a2
        eq(sorted_old, new)


# ---- A2 (port of the layout choice) ---------------------------------------------------------------------------------------
def _barcode_records(seed=5, n=6000):
    """Multiome-style 24 bp raw barcodes: 8 bp spacer + 16 bp whitelist barcode. The first 1500 records of the (position-sorted)
    stream are from a region with junk barcodes, as the start of chr1 can be."""
    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    wl = ["".join(rng.choice(bases, 16)) for _ in range(200)]
    recs = []
    for i in range(n):
        if i < 1500:
            raw = "".join(rng.choice(bases, 24))                      # no whitelist structure
        else:
            raw = "".join(rng.choice(bases, 8)) + wl[rng.integers(200)]
        recs.append((f"BC:{raw}_r{i}", raw))
    return recs, {w for w in wl}


def test_a2_hash_sample_choice_is_order_free_and_prefix_choice_is_not():
    recs, wl = _barcode_records()
    junk = {"".join(["A"] * 16)}
    ref = B.choose(B.hash_sample(recs), [junk, wl])
    assert ref[0] == 1 and ref[1][0] == 8 and not ref[1][1] and ref[1][2] > 0.7      # offset 8, forward, whitelist 1
    for s in range(3):
        sh = [recs[i] for i in np.random.default_rng(s).permutation(len(recs))]
        assert B.choose(B.hash_sample(sh), [junk, wl]) == ref
    # PR #48 rule with a prefix cap of 1000: the sorted order decides (junk first), a shuffled order does not
    pre_sorted = B.choose(B.prefix_sample(recs, 1000), [junk, wl])
    sh = [recs[i] for i in np.random.default_rng(0).permutation(len(recs))]
    pre_shuf = B.choose(B.prefix_sample(sh, 1000), [junk, wl])
    assert pre_sorted[1][2] < 0.1 and pre_shuf != pre_sorted


def test_a2_hash_is_about_one_in_64():
    n = sum(B.is_layout_sample(f"BC:AAAA_r{i}") for i in range(20000))
    assert 200 < n < 450


# ---- A8: peaks from called cells -------------------------------------------------------------------------------------------
def test_a8_final_peaks_come_from_called_cells_only(data):
    rec, planted = data
    out = R.run_all(rec, 1, sim.CHROM_LEN, sim.IS_MITO, sim.NCELLS)
    is_cell = out["is_cell"].astype(bool)
    assert is_cell[: sim.N_REAL].sum() >= 55 and not is_cell[sim.N_REAL:].any()      # empties are not called
    f = out["fragments"]
    cells_only = R.insertion_reduce(R.insertion_emit(R.select(f, is_cell[f["cell"]]), sim.IS_MITO))
    eq(R.peak_call(cells_only, sim.CHROM_LEN, sim.IS_MITO), out["peaks"])
    # the first pass (all barcodes) is a different insertion table: it holds the empty droplets' fragments
    assert out["insertions_all"]["count"].sum() > out["insertions"]["count"].sum()
    hit = sum(any(out["peaks"]["tid"][i] == t and out["peaks"]["start"][i] - 200 <= c <= out["peaks"]["end"][i] + 200
                  for i in range(len(out["peaks"]["tid"]))) for t, c in planted)
    assert hit >= 0.8 * len(planted)
    # no peak is built from a non-cell's cut sites: counts in the final matrix of non-cells are possible (they are counted against
    # the peaks), but they do not enter the peak set
    assert out["peak_counts"]["cell"].max() < sim.NCELLS

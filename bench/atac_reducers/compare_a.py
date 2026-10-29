"""Compare the mode A outputs of atac_reducers_test with the numba reference (ref.npz) and with the CPU's own fragments.tsv.gz rows.
  python compare_a.py MODEA_DIR GPU_OUT_DIR
"""
import sys
import numpy as np

d, g = sys.argv[1:3]
ref = np.load(f"{d}/ref.npz")
ok = True
def rep(name, same, extra=""):
    global ok
    ok &= bool(same)
    print(f"[cmp] {name:34s} {'EQUAL' if same else 'DIFFERENT'} {extra}")

fr = np.fromfile(f"{g}/frags.bin", np.uint32).reshape(-1, 5)
rf = np.stack([ref["frag_tid"], ref["frag_start"], ref["frag_end"], ref["frag_cell"], ref["frag_read_pairs"]], 1).astype(np.uint32)
rep("fragments vs numba reference", fr.shape == rf.shape and np.array_equal(fr, rf), f"gpu {len(fr)} ref {len(rf)}, read_pairs sum {int(fr[:, 4].sum())} / {int(rf[:, 4].sum())}")
print(f"[cmp] old per-start dedup, shuffled order: {len(ref['fold_sh_cell'])} fragments (reducer: {len(fr)}); old, BAM order: {len(ref['fold_cell'])}")

def ins(path):
    a = np.fromfile(path, np.uint8); n = int(np.frombuffer(a[:8].tobytes(), np.uint64)[0])
    return np.frombuffer(a[8:8 + 8 * n].tobytes(), np.uint64), np.frombuffer(a[8 + 8 * n:8 + 12 * n].tobytes(), np.uint32)
for tag, pre in (("insertions (all barcodes)", "insall"), ("insertions (called cells)", "ins")):
    k, c = ins(f"{g}/ins_all.bin" if pre == "insall" else f"{g}/ins.bin")
    rk = (ref[f"{pre}_tid"].astype(np.uint64) << np.uint64(32)) | ref[f"{pre}_pos"].astype(np.uint64)
    rep(tag + " vs reference", np.array_equal(k, rk) and np.array_equal(c, ref[f"{pre}_count"].astype(np.uint32)), f"gpu {len(k)} ref {len(rk)} entries, {int(c.sum())} / {int(ref[f'{pre}_count'].sum())} insertions")

pdt = np.dtype([("tid", "<i4"), ("start", "<i4"), ("end", "<i4"), ("summit", "<i4"), ("pileup", "<f4"), ("nlp", "<f4"), ("nlq", "<f4")])
for tag, pre, fn in (("peaks (all barcodes)", "peaksall", "peaks_all.bin"), ("peaks (A8, called cells)", "peaks", "peaks.bin")):
    p = np.fromfile(f"{g}/{fn}", pdt)
    same = len(p) == len(ref[f"{pre}_tid"]) and all(np.array_equal(p[k], ref[f"{pre}_{k}"].astype(p[k].dtype)) for k in pdt.names)
    extra = f"gpu {len(p)} ref {len(ref[pre + '_tid'])}"
    if not same and len(p) == len(ref[pre + "_tid"]):
        for k in pdt.names:
            nb = int((p[k] != ref[f"{pre}_{k}"].astype(p[k].dtype)).sum())
            if nb: extra += f"; {k}: {nb} differ"
    rep(tag + " vs reference", same, extra)

t = np.fromfile(f"{g}/triplets.bin", np.uint32).reshape(-1, 3)
rt = np.stack([ref["cnt_cell"], ref["cnt_peak"], ref["cnt_count"]], 1).astype(np.uint32)
rep("cell x peak triplets vs reference", t.shape == rt.shape and np.array_equal(t, rt), f"gpu {len(t)} ref {len(rt)} nonzeros, cut sites {int(t[:, 2].sum())} / {int(rt[:, 2].sum())}")
m = np.fromfile(f"{g}/metrics.bin", np.uint32).reshape(-1, 5)
rm = np.stack([ref["met_read_pairs"], ref["met_fragments"], ref["met_mito_fragments"], ref["met_peak_region_fragments"], ref["met_peak_region_cutsites"]], 1).astype(np.uint32)
rep("per-cell counters vs reference", np.array_equal(m, rm), "(read_pairs, fragments, mito, peak_region_fragments, peak_region_cutsites) cells with data: %d" % int((m[:, 0] > 0).sum()))

# the CPU's own fragments.tsv.gz rows (written by the unmodified engine with barcode correction) for the same contigs
chroms = [l.split("\t")[0] for l in open(f"{d}/chroms.tsv")]
cid = {c: i for i, c in enumerate(chroms)}
wl = {l.rstrip("\n"): i for i, l in enumerate(open(sys.argv[3]))} if len(sys.argv) > 3 else None
if wl:
    rows = [l.rstrip("\n").split("\t") for l in open(f"{d}/cpu_frags_sub.tsv")]
    ck = np.array([[cid[r[0]], int(r[1]), int(r[2]), wl[r[3]], int(r[4])] for r in rows], np.int64)
    def keyof(a): return (a[:, 0].astype(np.uint64) << np.uint64(60)) | (a[:, 1].astype(np.uint64) << np.uint64(40)) | (a[:, 2].astype(np.uint64) << np.uint64(20)) | a[:, 3].astype(np.uint64)
    # key packing: (tid 4+ bits, start 28 bits, end 28, cell 20) does not fit 64 bits; use a structured join instead
    cdt = [("t", "<i8"), ("s", "<i8"), ("e", "<i8"), ("c", "<i8")]
    A = np.rec.fromarrays([ck[:, 0], ck[:, 1], ck[:, 2], ck[:, 3]], dtype=cdt); B = np.rec.fromarrays([fr[:, 0].astype(np.int64), fr[:, 1].astype(np.int64), fr[:, 2].astype(np.int64), fr[:, 3].astype(np.int64)], dtype=cdt)
    ia, ib = np.argsort(A), np.argsort(B)
    Au, Bu = A[ia], B[ib]
    pos = np.searchsorted(Au, Bu)
    found = (pos < len(Au))
    found[found] = Au[pos[found]] == Bu[found]
    eq = np.zeros(len(Bu), bool); eq[found] = ck[ia][pos[found], 4] == fr[ib][found, 4]
    print(f"[cpu] CPU engine rows in these contigs: {len(ck)} (read_pairs {int(ck[:, 4].sum())}); GPU fragments {len(fr)} (read_pairs {int(fr[:, 4].sum())})")
    print(f"[cpu] GPU fragments present in the CPU output: {int(found.sum())} ({100.0 * found.mean():.3f}%); with equal read_pairs: {int(eq.sum())} ({100.0 * eq.mean():.3f}%); GPU-only: {int((~found).sum())}")
    print(f"[cpu] CPU rows not in the GPU output: {len(ck) - int(found.sum())} ({100.0 * (len(ck) - found.sum()) / len(ck):.3f}%), expected: fragments whose barcode the CPU corrected (Hamming-1) and the harness did not")
print("RESULT", "PASS" if ok else "FAIL")

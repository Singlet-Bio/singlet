"""Mode A inputs and references for atac_reducers_test: real alignment records of a few contigs of a kept ATAC BAM, and the numba reference
(atac_ref.py: sequential CPU ports and the reducer-shaped functions) computed from the same records.

  python prep_a.py RECORDS.txt CHROMS.tsv WHITELIST.txt OUTDIR
RECORDS.txt: one line per alignment record: bc tid pos0 tlen mtid flag mapq   (bc = whitelist id of the exact raw barcode, -1 if not an exact hit)
"""
import sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import atac_ref as R

txt, chroms_p, wl_p, out = sys.argv[1:5]
t0 = time.time()
def lap(m): print(f"[prep] {m} ({time.time() - t0:.0f} s)", flush=True)
chroms = [l.rstrip("\n").split("\t") for l in open(chroms_p)]
chrom_len = np.array([int(c[1]) for c in chroms], np.int64)
is_mito = np.array([c[0] in ("chrM", "MT", "chrMT", "M", "mitochondrion_genome") for c in chroms])
wl = [l.rstrip("\n") for l in open(wl_p)]
ncells = len(wl)
a = np.loadtxt(txt, dtype=np.int64)
lap(f"loaded {len(a)} records")
dt = np.dtype([("bc", "<i4"), ("tid", "<i4"), ("pos", "<i4"), ("tlen", "<i4"), ("mtid", "<i4"), ("flag", "<u2"), ("mapq", "u1"), ("pad", "u1")])
ra = np.zeros(len(a), dt)
for i, k in enumerate(["bc", "tid", "pos", "tlen", "mtid", "flag", "mapq"]):
    ra[k] = a[:, i]
ra.tofile(f"{out}/recs.bin")
rec = {"cell": a[:, 0], "tid": a[:, 1], "pos": a[:, 2], "tlen": a[:, 3], "mtid": a[:, 4], "flag": a[:, 5], "mapq": a[:, 6]}
print("records: total %d, resolved barcode %d, MAPQ>=30 %d, proper %d" % (len(a), (a[:, 0] >= 0).sum(), (a[:, 6] >= 30).sum(), ((a[:, 5] & 2) > 0).sum()))

# CPU ports on the records in BAM (coordinate) order; and on a shuffled order (finding F1)
f_old = R.cpu_fragments(rec, assume_sorted=True); lap(f"cpu port, per-start dedup table (old default), BAM order: {len(f_old['cell'])} fragments, read_pairs {f_old['read_pairs'].sum()}")
f_a1 = R.cpu_fragments(rec, assume_sorted=False); lap(f"cpu port, global dedup table (A1), BAM order: {len(f_a1['cell'])} fragments")
perm = np.random.default_rng(1).permutation(len(a)); rec_sh = {k: v[perm] for k, v in rec.items()}
f_old_sh = R.cpu_fragments(rec_sh, assume_sorted=True); lap(f"cpu port, old dedup, shuffled order: {len(f_old_sh['cell'])} fragments")
f_a1_sh = R.cpu_fragments(rec_sh, assume_sorted=False); lap(f"cpu port, A1, shuffled order: {len(f_a1_sh['cell'])} fragments")
f = R.coord_order(R.fragment_reduce(R.fragment_emit(rec))); lap(f"reducer reference: {len(f['cell'])} fragments")
same = lambda x, y: all(np.array_equal(x[k], y[k]) for k in ("tid", "start", "end", "cell", "read_pairs"))
print("[prep] reference reducer == cpu port old(sorted):", same(f, f_old), " == cpu port A1(sorted):", same(f, f_a1), " == A1(shuffled):", same(f, f_a1_sh), " old(shuffled) differs:", not same(f, f_old_sh))

ins_all = R.insertion_reduce(R.insertion_emit(f, is_mito)); lap(f"insertions {len(ins_all['pos'])}")
peaks_all = R.peak_call(ins_all, chrom_len, is_mito); lap(f"peaks (all barcodes) {len(peaks_all['tid'])}")
m_all = R.cell_metrics(f, peaks_all, is_mito, ncells)
is_cell = R.default_cell_caller(m_all)
open(f"{out}/cells.txt", "w").write("".join(wl[i] + "\n" for i in np.flatnonzero(is_cell)))
fc = R.select(f, is_cell[f["cell"]])
ins = R.insertion_reduce(R.insertion_emit(fc, is_mito)); peaks = R.peak_call(ins, chrom_len, is_mito); lap(f"peaks (A8, {int(is_cell.sum())} called cells) {len(peaks['tid'])}")
cnt = R.peak_count_reduce(R.peak_count_emit(f, peaks, is_mito))
met = R.cell_metrics(f, peaks, is_mito, ncells)
np.savez(f"{out}/ref.npz",
         **{f"frag_{k}": v for k, v in f.items()}, **{f"fold_{k}": v for k, v in f_old.items()}, **{f"fold_sh_{k}": v for k, v in f_old_sh.items()},
         **{f"insall_{k}": v for k, v in ins_all.items()}, **{f"ins_{k}": v for k, v in ins.items()},
         **{f"peaksall_{k}": v for k, v in peaks_all.items()}, **{f"peaks_{k}": v for k, v in peaks.items()},
         **{f"cnt_{k}": v for k, v in cnt.items()}, **{f"met_{k}": v for k, v in met.items()}, is_cell=is_cell)
lap("saved")

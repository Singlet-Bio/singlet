"""Is a run identical to a reference run? Matrices (perf_equiv.compare), STAR SJ.out.tab, Solo
outputs, Log.final.out counts, pileup stats, and optionally the tagged BAM record streams.

    python equiv.py REF_DIR RUN_DIR [RUN_DIR...]
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/jobs/atlas-rerun")
from perf_equiv import MEMBERS, compare, load  # noqa: E402

TEXT = ["star_SJ.out.tab", "star_Solo.out/Barcodes.stats", "star_Solo.out/Gene/Features.stats",
        "star_Solo.out/Gene/Summary.csv", "star_Solo.out/Gene/UMIperCellSorted.txt",
        "star_Solo.out/Gene/raw/matrix.mtx", "star_Solo.out/Gene/raw/barcodes.tsv",
        "star_Solo.out/Gene/filtered/matrix.mtx", "star_Solo.out/Gene/filtered/barcodes.tsv",
        "cell_calls.tsv", "cell_qc_metrics.tsv", "doublet_scores.tsv", "mt_variants.tsv", "splice_events.tsv"]


def digest(p: Path):
    if not p.exists():
        return None
    h = hashlib.sha1()
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def final_counts(p: Path):
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if "|" in line and not any(k in line for k in ("Started", "Finished", "speed")):
                k, v = line.split("|", 1)
                out[k.strip()] = v.strip()
    return out


def bam_cmp(a: Path, b: Path):
    sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/envs/pylib-pysam")
    import pysam
    fa, fb = pysam.AlignmentFile(str(a), "rb"), pysam.AlignmentFile(str(b), "rb")
    ha = [x for x in str(fa.header).splitlines() if not x.startswith(("@PG", "@CO"))]
    hb = [x for x in str(fb.header).splitlines() if not x.startswith(("@PG", "@CO"))]
    n = diff = 0; first = None
    for ra, rb in zip(fa, fb):
        n += 1
        if ra.to_string() != rb.to_string():
            diff += 1
            if first is None:
                first = (ra.to_string()[:300], rb.to_string()[:300])
    extra = sum(1 for _ in fa) + sum(1 for _ in fb)
    return {"records": n, "differing": diff, "extra_records": extra, "header_equal": ha == hb, "first_diff": first}


def main():
    ref = Path(sys.argv[1])
    ref_m = {m: load(ref, m) for m in MEMBERS}
    ref_t = {t: digest(ref / t) for t in TEXT}
    ref_f = final_counts(ref / "star_Log.final.out")
    ref_s = json.loads((ref / "pileup_stats.json").read_text())
    for run in map(Path, sys.argv[2:]):
        res = {}
        for m in MEMBERS:
            b = load(run, m)
            res[m] = compare(ref_m[m], b) if (b is not None and ref_m[m] is not None) else "missing"
        txt = {t: (ref_t[t] == digest(run / t)) for t in TEXT if ref_t[t] is not None}
        fc = final_counts(run / "star_Log.final.out")
        fdiff = {k: (ref_f.get(k), fc.get(k)) for k in set(ref_f) | set(fc) if ref_f.get(k) != fc.get(k)}
        st = json.loads((run / "pileup_stats.json").read_text()) if (run / "pileup_stats.json").exists() else {}
        sdiff = {k: (ref_s.get(k), st.get(k)) for k in set(ref_s) | set(st)
                 if ref_s.get(k) != st.get(k) and not any(w in k for w in ("time", "sec", "_s"))}
        mat_ok = all(isinstance(v, dict) and v["identical"] for v in res.values())
        ok = mat_ok and all(txt.values()) and not fdiff
        print(f"{run}: {'IDENTICAL' if ok else 'DIFFERS'}  matrices={'same' if mat_ok else 'DIFF'}", flush=True)
        for m, v in res.items():
            if not (isinstance(v, dict) and v["identical"]):
                print(f"   {m}: {v if not isinstance(v, dict) else {k: v[k] for k in ('entries_differing', 'abs_diff', 'total_ratio_on_shared', 'barcodes_ref', 'barcodes_run')}}")
        for t, same in txt.items():
            if not same:
                print(f"   text differs: {t}")
        if fdiff:
            print(f"   Log.final.out: {fdiff}")
        if sdiff:
            print(f"   pileup_stats: {sdiff}")
        ba, bb = ref / "possorted_genome_bam.bam", run / "possorted_genome_bam.bam"
        if ba.exists() and bb.exists():
            print(f"   BAM: {bam_cmp(ba, bb)}", flush=True)


if __name__ == "__main__":
    main()

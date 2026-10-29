"""Local plumbing test for run_class.py without a GPU: a pure-Python stand-in for bl_harness (built on the reference),
synthetic reads, and a fake SAM in place of STAR's BAM. It checks file formats and the comparison code, not the CUDA kernels."""
import os, random, subprocess, sys, tempfile, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "src" / "engine_gpu"))
import numpy as np, bl_io as io, barcode_layer_ref as ref


def fake_harness(argv):
    a = dict(x.split("=", 1) for x in argv if "=" in x)
    off, seq, qual = io.load_reads(a["reads"])
    n = len(off) - 1
    out = a["out"]
    def w(name, typ, arr):
        dt = {"i64": "<i8", "i32": "<i4", "u64": "<u8", "u32": "<u4"}[typ]
        np.asarray(arr, dtype=dt).tofile(f"{out}.{name}.{typ}")
    S = lambda i: seq[off[i]:off[i + 1]].tobytes().decode()
    Q = lambda i: qual[off[i]:off[i + 1]].tobytes().decode()
    if a["mode"] == "simple":
        keys = np.fromfile(a["wl"], dtype="<u8")[1:]
        cb0, cbl, um0, uml = (int(a[k]) for k in ("cb_start", "cb_len", "umi_start", "umi_len"))
        lay = ref.SimpleLayout(cb0 + 1, cbl, um0 + 1, uml, ref.flags_from_type("1MM_multi_Nbase_pseudocounts"), ref.Whitelist_from_sorted(list(keys)))
        rbs = [ref.get_cb_umi_simple(S(i), Q(i), lay) for i in range(n)]
        counts = ref.exact_counts(rbs, len(keys), lay.flags)
        he = counts - 1
        cell, st, um, cv, pn = [], [], [], [], []
        for i, rb in enumerate(rbs):
            c, why = ref.resolve_cb(rb, counts, lay.flags) if rb.cb_match >= 0 else (-1, "")
            cell.append(c); um.append(ref.pack3(rb.umi_seq)); v, p = ref.conv(rb.cb_seq); cv.append(v); pn.append(p)
            if rb.cb_match < 0: st.append(rb.cb_match)
            elif c < 0: st.append(-4)
            else: st.append(min(rb.cb_match, 2))
        w("cell", "i64", cell); w("status", "i32", st); w("umi", "u64", um); w("umilen", "u32", [uml] * n); w("cbval", "u64", cv); w("posn", "i32", pn); w("histexact", "u32", he)
    elif a["mode"] == "complex":
        raise SystemExit("complex not in selftest")
    else:
        raise SystemExit("mode not in selftest")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "run":
        fake_harness(sys.argv[1:]); sys.exit(0)
    rng = random.Random(5)
    d = tempfile.mkdtemp()
    os.makedirs(f"{d}/data/pbmc1k_v3"); os.makedirs(f"{d}/wl")
    wl = sorted({''.join(rng.choice("ACGT") for _ in range(16)) for _ in range(3000)})
    with open(f"{d}/wl/3M-february-2018.txt", "w", newline="\n") as f: f.write("\n".join(wl) + "\n")
    # a few close pairs so the posterior is exercised
    base = wl[0]
    n = 20000
    with open(f"{d}/data/pbmc1k_v3/R1.fq", "w", newline="\n") as f:
        for i in range(n):
            cb = list(rng.choice(wl) if rng.random() > 0.2 else wl[rng.randrange(20)])
            r = rng.random()
            if r < 0.08: p = rng.randrange(16); cb[p] = rng.choice("ACGT")
            elif r < 0.10: cb[rng.randrange(16)] = "N"
            umi = ''.join(rng.choice("ACGT") for _ in range(12))
            if rng.random() < 0.01: umi = "A" * 12
            s = ''.join(cb) + umi
            f.write(f"@r{i}\n{s}\n+\n{''.join(rng.choice('FF:,#') for _ in s)}\n")
    # fake STAR SAM: CB = what an independent pass says, with a few edits
    cmd = [sys.executable, "-P", str(HERE / "run_class.py"), "--cls", "pbmc1k_v3_real", "--exe", f"{sys.executable} {HERE / 'selftest.py'}", "--work", f"{d}/w", "--out", f"{d}/o",
           "--data", f"{d}/data", "--wl-dir", f"{d}/wl", "--py-subset", "3000"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-600:]); print(r.stderr[-3000:])
    # fake STAR SAM from the stand-in's own output, with edits: drops, a wrong cell, a 1-base UB, unmapped flags
    work = f"{d}/w"; keys = np.fromfile(f"{work}/wl.bin", dtype="<u8")[1:]
    cell = io.load_arr(f"{work}/gpu", "cell", "i64"); umi = io.load_arr(f"{work}/gpu", "umi", "u64")
    cb = io.decode2(keys[np.maximum(cell, 0)], 16); um = io.decode3(umi, 12)
    T, NL = chr(9), chr(10)
    with open(f"{d}/x.sam", "w", newline=NL) as f:
        for i in range(len(cell)):
            if i % 97 == 0: continue                                   # no record
            fl = 4 if i % 5 == 0 else 0
            c = cb[i].decode() if cell[i] >= 0 else "-"
            if i % 211 == 0: c = cb[(i + 1) % len(cell)].decode()        # a different cell
            u = um[i].decode()
            ub = u if i % 7 else u[:-1] + ("A" if u[-1] != "A" else "C")
            ubv = ub if cell[i] >= 0 else "-"
            f.write(T.join([f"r{i}", str(fl), "*", "0", "0", "*", "*", "0", "0", "N", "F", "CR:Z:x", f"UR:Z:{u}", f"CB:Z:{c}", f"UB:Z:{ubv}"]) + NL)
    r = subprocess.run(cmd + ["--star-bam", f"{d}/x.sam"], capture_output=True, text=True)
    j = json.loads(r.stdout[r.stdout.index("{"):]) if "{" in r.stdout else None
    print(json.dumps(j["star"], indent=1) if j else r.stdout[-2000:] + r.stderr[-2000:])

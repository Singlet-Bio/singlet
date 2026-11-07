#!/usr/bin/env python3
"""Compare the --engine gpu (GPU pileup) outputs of the release gate with the CPU engine's, per dataset.

    python gpu_diff.py CPU_ROOT_FMT GPU_ROOT OUT.json     (CPU_ROOT_FMT has {ds}, e.g. .../runs/{ds}/singlet-public)

Every data output (matrices, barcodes, features, tables; not logs, timings or BAMs) is compared by the SHA-256 of its decompressed
bytes. Differing .mtx files are re-read and summarised (entries differing, total UMI difference). Each GPU run is classified from its
log: "GPU pileup" (the '[gpu_pileup] N reads' line) or "GPU engine, CPU pileup (fallback)" (the 'gpu_pileup: not run' / 'ignored' lines).
"""
import gzip, hashlib, json, os, sys, re

DS = ["pbmc1k_v3", "pbmc1k_v2", "neuron1k_v3", "hgmm1k_v3", "pbmc1k_cite_v3", "nuclei2k_v2", "pbmc5p_v1", "kang2018_ctrl"]
SKIP = re.compile(r"(\.log|\.txt|time|\.bam|\.bai|node|command|binary|stream_path|gpu_pileup_note|\.json|\.fq|\.1fq|STARtmp|_Log|\.progress)", re.I)


def opn(p):
    return gzip.open(p, "rb") if p.endswith(".gz") else open(p, "rb")


def digest(p):
    h = hashlib.sha256()
    with opn(p) as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def mtx(p):
    import numpy as np
    d = {}
    with opn(p) as f:
        for line in f:
            if line.startswith(b"%"):
                continue
            t = line.split()
            if len(t) == 3:
                if int(float(t[2])) != 0 or True:
                    pass
                d[(int(t[0]), int(t[1]))] = float(t[2])
    return d


def classify(log):
    t = open(log, errors="replace").read()
    if "gpu_pileup: not run" in t or "GPU_PILEUP ignored" in t:
        return "GPU engine, CPU pileup (fallback)"
    if re.search(r"^\[gpu_pileup\] \d+ reads", t, re.M):
        return "GPU pileup"
    return "unknown (no gpu_pileup line)"


def main(fmt, gpu_root, out):
    res = {}
    for ds in DS:
        c, g = fmt.format(ds=ds), os.path.join(gpu_root, ds)
        if not (os.path.isdir(c) and os.path.isdir(g) and os.path.exists(g + "/run.log")):
            res[ds] = {"status": "missing"}; continue
        r = {"gpu_run": classify(g + "/run.log"), "identical": [], "different": [], "only_cpu": [], "only_gpu": []}
        cf = {f for f in os.listdir(c) if os.path.isfile(os.path.join(c, f)) and not SKIP.search(f)}
        gf = {f for f in os.listdir(g) if os.path.isfile(os.path.join(g, f)) and not SKIP.search(f)}
        r["only_cpu"], r["only_gpu"] = sorted(cf - gf), sorted(gf - cf)
        for f in sorted(cf & gf):
            if digest(os.path.join(c, f)) == digest(os.path.join(g, f)):
                r["identical"].append(f)
            else:
                e = {"file": f}
                if ".mtx" in f:
                    a, b = mtx(os.path.join(c, f)), mtx(os.path.join(g, f))
                    keys = set(a) | set(b)
                    nd = sum(1 for k in keys if a.get(k, 0) != b.get(k, 0))
                    e.update(entries_differing=nd, entries=len(keys), sum_cpu=sum(a.values()), sum_gpu=sum(b.values()))
                r["different"].append(e)
        r["n_identical"], r["n_different"] = len(r["identical"]), len(r["different"])
        r["identical"] = r["identical"][:3] + ["..."] if len(r["identical"]) > 3 else r["identical"]
        res[ds] = r
        print(ds, r["gpu_run"], r["n_identical"], "identical", r["n_different"], "different", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])

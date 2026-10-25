"""Fetch the samples of samples.tsv for one array chunk (studies i::n) with the empty-droplet fetcher. Run inside rt."""
import importlib.util, os, sys, time
import pandas as pd
chunk, n, out, edrop, tsv = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
spec = importlib.util.spec_from_file_location("fetch", f"{edrop}/bench/empty_droplets/fetch.py")
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
d = pd.read_csv(tsv, sep="\t")
gses = sorted(d.gse.unique())[chunk::n]
cache = os.environ["WORK"] + "/data/emptydrop"
for gse in gses:
    gsms = [g for g in d[d.gse == gse].gsm if not os.path.exists(f"{out}/{gse}/{g}.npz") and not os.path.exists(f"{cache}/{gse}/{g}.npz")]
    if not gsms: continue
    t = time.time()
    try:
        F.main([gse, "--gsm", *gsms, "--out", out])
        missing = [g for g in gsms if not os.path.exists(f"{out}/{gse}/{g}.npz")]
        if missing: print(f"{gse} MISSING {missing}", flush=True)
    except Exception as e:
        print(f"{gse} STUDY_FAILED {type(e).__name__}: {e}", flush=True)
    print(f"{gse} done {time.time()-t:.0f}s", flush=True)

"""Ancestry proportions q for (a) the 1000 Genomes reference individuals that are not benchmark test individuals (exact genotypes,
leave-one-out frequencies, a fixed random subset of the panel) and (b) the 150 MAGE test individuals at 0.5/2/8M read pairs (RNA depth
tables, as ancestry_score.py). Writes only proportions: ref_q.tsv, test_q.tsv. Run on Clipper with PYTHONPATH = the repo's python/."""
import subprocess, sys, os
from multiprocessing import Pool
import numpy as np, pandas as pd
import importlib.util
_sp = importlib.util.spec_from_file_location('ancestry', os.environ['ANC_PY']); A = importlib.util.module_from_spec(_sp); sys.modules['ancestry'] = A; _sp.loader.exec_module(A)

BP = "/mnt/projects/debruinz_project/debruinz/singlet/bpub"; W = os.path.dirname(BP)
WA = BP + "/work/ancestry"; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
NSITES = 300_000
af = pd.read_parquet(WA + "/af_leaveout.parquet")
af["chrom"] = af["chrom"].astype(str).str.replace("^chr", "", regex=True)
groups = pd.read_csv(WA + "/groups.txt", sep="\t", header=None, names=["sample", "super_pop"])
_V = W + "/data/1kgp_grch38/ALL.chr22.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz"
VNAMES = subprocess.run(["bcftools", "query", "-l", _V], capture_output=True, text=True, check=True).stdout.split()
groups = groups[groups["sample"].isin(set(VNAMES))].reset_index(drop=True)  # a panel sample can be missing from the VCF
COL = [VNAMES.index(x) for x in groups["sample"]]  # bcftools prints samples in header order, whatever the -S order
print("reference individuals in the VCF", len(groups), flush=True)
panel = pd.read_csv(W + "/data/1kgp_grch38/panel.txt", sep="\t").set_index("sample")
POPS = A.POPS

def ref_part():
    rng = np.random.default_rng(1)
    sites = af.iloc[np.sort(rng.choice(len(af), NSITES, replace=False))].reset_index(drop=True)
    samples = groups["sample"].tolist()
    dos = np.full((len(sites), len(samples)), -1, np.int8)
    key = {(c, p): i for i, (c, p) in enumerate(zip(sites["chrom"], sites["pos"]))}
    ref, alt = sites["ref"].to_numpy(), sites["alt"].to_numpy()
    open(OUT + "/samples.txt", "w").write("\n".join(samples) + "\n")
    for c in sorted(set(sites["chrom"]), key=lambda x: int(x)):
        sub = sites[sites["chrom"] == c]
        reg = f"{OUT}/reg.{c}.tsv"; sub[["chrom", "pos"]].to_csv(reg, sep="\t", header=False, index=False)
        vcf = f"{W}/data/1kgp_grch38/ALL.chr{c}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz"
        cmd = ["bcftools", "query", "-R", reg, "-f", "%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n", vcf]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        for line in p.stdout:
            f = line.rstrip("\n").split("\t"); i = key.get((f[0].replace("chr", ""), int(f[1])))
            if i is None: continue
            g = np.array([x.count("1") if x[0] in "01" and x[2] in "01" else -1 for x in (f[4 + j] for j in COL)], np.int8)
            if (f[2], f[3]) == (ref[i], alt[i]): dos[i] = g
            elif (f[2], f[3]) == (alt[i], ref[i]): dos[i] = np.where(g >= 0, 2 - g, -1)
        p.wait(); print("chr", c, "done", flush=True)
    return sites, samples, dos

def fit_one(args):
    j, = args
    sp = groups["super_pop"].iloc[j]; k = POPS.index(sp)
    d = DOS[:, j].astype(np.int64); ok = d >= 0
    F = SITES[list(POPS)].to_numpy(np.float64)[ok].copy(); d = d[ok]
    n = NK[k]  # individuals of the own super-population in the AF table
    F[:, k] = np.clip((F[:, k] * 2 * n - d) / (2 * n - 2), 0, 1)  # leave this individual out
    L = A._site_terms(d * 10, np.full(len(d), 20), 0.5)
    q = A.fit_q(F, L)
    return [groups["sample"].iloc[j], sp, panel.loc[groups["sample"].iloc[j], "pop"]] + [float(x) for x in q]

def init(dos, sites, nk):
    global DOS, SITES, NK; DOS, SITES, NK = dos, sites, nk

if __name__ == "__main__":
    sites, samples, dos = ref_part()
    nk = [int((groups["super_pop"] == p).sum()) for p in POPS]
    print("reference individuals per super-pop", dict(zip(POPS, nk)), flush=True)
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)), initializer=init, initargs=(dos, sites, nk)) as pool:
        rows = pool.map(fit_one, [(j,) for j in range(len(samples))], chunksize=8)
    pd.DataFrame(rows, columns=["sample", "super_pop", "pop"] + ["q_" + p for p in POPS]).to_csv(OUT + "/ref_q.tsv", sep="\t", index=False)
    # test individuals: RNA depth tables
    subset = pd.read_csv("/mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__bench-public-cohort-485cf7/bench/public/sexancestry/mage_subset.tsv", sep="\t")
    idx = A.AFIndex(af); rows = []
    for ind in subset.iloc[:, 1]:
        for L_ in ["0.5", "2", "8"]:
            p = f"{WA}/mage/private/{ind}/{L_}/sites.parquet"
            if not os.path.exists(p): continue
            ad, dp, F = idx.align(pd.read_parquet(p))
            c = A.infer_ancestry(ad, dp, F, n_boot=0)
            rows.append([ind, L_, c.group, c.n_sites] + [c.q.get(p_, np.nan) for p_ in POPS])
    pd.DataFrame(rows, columns=["individual", "level", "call", "n_sites"] + ["q_" + p for p in POPS]).to_csv(OUT + "/test_q.tsv", sep="\t", index=False)
    print("done")

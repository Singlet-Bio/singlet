"""Write provenance.json for bench/public/capabilities/doublets.py from the measured job (bench/doublets/e2e.sbatch).
Usage: make_provenance.py OUT_DIR BENCH_COMMIT PARITY_JSON SIDECAR_JSON_1 [...]"""
import json
import sys
from datetime import datetime, timezone

out, commit, parity, *sides = sys.argv[1:]
par = json.load(open(parity))
runs = [json.load(open(p)) for p in sides]
r0 = runs[0]
methods = (
    "Data: 17 datasets with experimental doublet labels: Kang 2018 demuxlet genotype calls (lane 2.1 as Singlet counts it from the "
    "Singlet pipeline, and the five lanes of Xi & Li 2021 as Cell Ranger counts), cell hashing, MULTI-seq and human-mouse species "
    "mixing (Xi & Li 2021, Zenodo 4062232, public count matrices). Only doublets visible to the experiment are labelled. "
    "Singlet: legacy_doublets, a gradient-boosted ensemble (300 trees, 15 leaves) over the within-sample ranks of log UMI, log "
    "detected genes, the engine's doublet score (doublet_detect.h v7) and four features of the atlas cell-type network (k-NN "
    "fraction of simulated doublets in the network's embedding, cross-fitted real-versus-simulated classifier, cell-type entropy, "
    "two-centroid fit). The ensemble scoring a dataset was trained on the labels of every dataset of the other studies "
    "(leave-one-study-out; the six Kang lanes are one study, the five MULTI-seq sets another, the two 10x hgmm sets a third), each "
    "dataset weighing the same, so no number here is on cells the model saw. Competitors: the previous Singlet scorer alone, "
    "scDblFinder 1.23 and Scrublet 0.2.3 with defaults, on the same counts; no simulated doublet is part of the evaluation. "
    "Metrics: AUPRC and AUROC of each tool's score against the labels, per dataset; pooled values are means over datasets (or "
    "over studies) with a bootstrap over datasets. Pooled metrics gate against the best competitor with a tolerance equal to the "
    "paired bootstrap half-width; per-dataset AUPRC gates against the previous Singlet scorer. Cost: seconds per 10,000 cells from "
    "the study's timings, and for Singlet from the batch job on Kang 2.1 (one V100S GPU, 8 CPU threads, including the engine "
    "score). Genotype doublets (prob_doublet) are not compared here; they stay the call for pools with two or more donors."
)
prov = dict(
    bench_commit=commit, status="draft", generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    singlet=dict(version="1.0.0", commit=commit, branch="claude/doublet-score",
                 binary="python: legacy_doublets (combiner gbm300-17ds-1) with dbl_run from prod/modality-fixes 4f110f283cd4"),
    methods=methods,
    cost=dict(singlet=round(par["timings"]["total_s"] / par["cells"] * 1e4, 1), prod=2.0, scdblfinder=43.6, scrublet=7.7),
    resources=[dict(tool="singlet", dataset="pooled", wall_s=r0["timings"]["total_s"], max_rss_gb=r0["max_rss_gb"], threads=8,
                    notes=f"{r0['gse']} {r0['gsm']}: {r0['n_cells']} cells, one V100S GPU, includes the engine score "
                          f"({r0['timings']['prod_s']} s) and CUDA start-up")],
    caveats=[f"Parity of the batch job with the study's cached features on Kang 2.1: Spearman "
             + ", ".join(f"{k} {v}" for k, v in par["spearman_vs_cached"].items()) + "."],
)
json.dump(prov, open(f"{out}/provenance.json", "w"), indent=1)
print(len(methods.split()), "methods words")

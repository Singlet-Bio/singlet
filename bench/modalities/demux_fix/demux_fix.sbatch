#!/bin/bash
#SBATCH --job-name=demux-fix
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Donor-demux fix on the Kang 8-donor pool: compiles demux_harness.cpp against the donor_demux.h that was put in
# $J/inc, loads singlet_k8/donor/snp_{ad,dp}.mtx.gz once, runs each configuration in RUNS on the called cells and
# scores every run (eval_demux.py: ARI, doublet AUROC/AUPRC) next to vireo.
#   sbatch --export=ALL,LABEL=try1,RUNS="k8 8 -- auto -1" demux_fix.sbatch
# RUNS = "TAG K [key=value ...] -- TAG K [key=value ...]"  (K -1 = auto); keys are listed in demux_harness.cpp.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/modfix-demux; G=$W/modality-bench/genotype_pool
LABEL=${LABEL:-try}; O=$W/modality-bench/demux_fix/$LABEL
PY=$W/envs/gpool/venv/bin/python
export PATH=$W/envs/pbench/bin:$PATH
mkdir -p $O
x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -pthread -I$J/inc $J/demux_harness.cpp -o $O/demux_harness
D=$G/singlet_k8/donor
[ -f $W/modality-bench/demux_fix/subsets/d6.txt ] || (cd $J && $PY make_subsets.py $G/truth/GSE96583_batch2.total.tsne.df.tsv.gz $W/modality-bench/demux_fix/subsets)
/usr/bin/time -v -o $O/time.txt $O/demux_harness $D/snp_ad.mtx.gz $D/snp_dp.mtx.gz $D/snp_ad_barcodes.tsv.gz \
  $G/singlet_k8/cell_calls.tsv $O ${SLURM_CPUS_PER_TASK:-16} $RUNS
grep -E "Elapsed|Maximum resident" $O/time.txt || true

# score: every run directory with a donor_assignments.tsv, plus vireo (the reference)
ARGS=""
for d in $O/*/; do
  t=$(basename $d)
  [ -f $d/donor_assignments.tsv ] && ARGS="$ARGS singlet_$t=$d/donor_assignments.tsv"
done
cd $J
$PY eval_demux.py $G/truth/GSE96583_batch2.total.tsne.df.tsv.gz $O/eval.json $ARGS vireo=$G/ref_k8/vireo/donor_ids.tsv > $O/eval.txt
$PY - <<EOF
import json
r = json.load(open("$O/eval.json"))
print("%-22s %7s %7s %7s %6s %6s %7s %7s" % ("run", "ARI", "ARIasg", "unasg", "sdbl", "nDon", "AUROC", "AUPRC"))
for k, v in r.items():
    print("%-22s %7.3f %7.3f %7.4f %6.4f %6d %7.3f %7.3f" % (k, v["ari_all_singlets"], v["ari_assigned_only"],
        v["unassigned_rate"], v["singlets_called_doublet"], v["n_donors_found"], v["dbl_auroc_genotype"], v["dbl_auprc_genotype"]))
EOF
for d in $O/*/; do [ -f $d/summary.json ] && cat $d/summary.json; done

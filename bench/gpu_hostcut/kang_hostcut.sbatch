#!/bin/bash
#SBATCH --job-name=ghc-kang
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# gpu/bold-hostcut: the real Kang 8-donor pool must still give auto-K 8 with the single-donor screen on. Harness: bench/modalities/demux_fix
# (demux_harness.cpp, eval_demux.py). The harness is compiled here against include/ of the branch (BR, default gpu/bold-hostcut; EXPECT is an optional
# commit prefix that the branch tip must have). Runs on the called cells of singlet_k8: auto-K with the screen on and off. Gate: screen-on auto-K = 8,
# assignments identical to screen-off. CPU only.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/modfix-demux; G=$W/modality-bench/genotype_pool
BR=${BR:-gpu/bold-hostcut}; EXPECT=${EXPECT:-}
O=$W/jobs/gpu-hostcut/results/kang-$SLURM_JOB_ID; mkdir -p $O
PY=$W/envs/gpool/venv/bin/python
export PATH=$W/envs/pbench/bin:$PATH
BARE=$W/repos/singlet-py.git; URL="$(git -C $BARE config remote.origin.url)"
WANT=$(git ls-remote "$URL" refs/heads/$BR | cut -f1); [ -n "$WANT" ] || { echo "no branch $BR"; exit 3; }
[ -z "$EXPECT" ] || [[ $WANT == $EXPECT* ]] || { echo "branch tip $WANT, EXPECT is $EXPECT"; exit 3; }
SREF=refs/ghc/kang-$SLURM_JOB_ID
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$BR:$SREF ) 9>$BARE/.gn-fetch.lock
[ "$(git -C $BARE rev-parse $SREF)" = "$WANT" ] || { echo "fetched another commit than the tip $WANT"; exit 3; }
SCR=/tmp/$USER-ghck-$SLURM_JOB_ID; mkdir -p $SCR; trap 'git -C $BARE update-ref -d '$SREF' 2>/dev/null; rm -rf "${SCR:?}"' EXIT
git -C $BARE archive $SREF include bench/modalities/demux_fix | tar -x -C $SCR
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) branch=$BR commit=$WANT $(date -Is)" | tee $O/summary.txt
x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -pthread -I$SCR/include $SCR/bench/modalities/demux_fix/demux_harness.cpp -o $O/demux_harness
D=$G/singlet_k8/donor; export CALLS_COL=is_cell
/usr/bin/time -v -o $O/time.txt $O/demux_harness $D/snp_ad.mtx.gz $D/snp_dp.mtx.gz $D/snp_ad_barcodes.tsv.gz \
  $G/singlet_k8/cell_calls.tsv $O ${SLURM_CPUS_PER_TASK:-16} auto_on -1 screen_single_donor=1 -- auto_off -1 screen_single_donor=0 2>&1 | tee $O/harness.log | grep -a "screen\|auto-K: selected\|Best K" | cut -c1-200
grep -E "Elapsed|Maximum resident" $O/time.txt | tee -a $O/summary.txt
ARGS=""
for d in $O/*/; do t=$(basename $d); [ -f $d/donor_assignments.tsv ] && ARGS="$ARGS singlet_$t=$d/donor_assignments.tsv"; done
cd $J
$PY eval_demux.py $G/truth/GSE96583_batch2.total.tsne.df.tsv.gz $O/eval.json $ARGS vireo=$G/ref_k8/vireo/donor_ids.tsv > $O/eval.txt
$PY - <<EOF | tee -a $O/summary.txt
import json
r = json.load(open("$O/eval.json"))
print("%-22s %7s %7s %7s %6s %6s %7s %7s" % ("run", "ARI", "ARIasg", "unasg", "sdbl", "nDon", "AUROC", "AUPRC"))
for k, v in r.items():
    print("%-22s %7.3f %7.3f %7.4f %6.4f %6d %7.3f %7.3f" % (k, v["ari_all_singlets"], v["ari_assigned_only"],
        v["unassigned_rate"], v["singlets_called_doublet"], v["n_donors_found"], v["dbl_auroc_genotype"], v["dbl_auprc_genotype"]))
on = r.get("singlet_auto_on", {}).get("n_donors_found")
print("KANG_AUTOK_SCREEN_ON", on, "PASS" if on == 8 else "FAIL")
EOF
if cmp -s $O/auto_on/donor_assignments.tsv $O/auto_off/donor_assignments.tsv; then echo "KANG_ASSIGNMENTS screen on == screen off IDENTICAL" | tee -a $O/summary.txt
else echo "KANG_ASSIGNMENTS screen on vs off DIFFER" | tee -a $O/summary.txt; fi
for d in $O/auto_on $O/auto_off; do [ -f $d/summary.json ] && { echo $d; cat $d/summary.json; } | tee -a $O/summary.txt; done
echo done $(date -Is)

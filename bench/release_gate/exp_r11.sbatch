#!/bin/bash
#SBATCH --job-name=rg-r11
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Task 4: why did hgmm1k_v3 umi_ratio_distance.total move from 0.0089 to 0.0169 (0.0177 at the pin) with #80 / R11?
# Same input and flags as counts_singlet.sbatch for hgmm1k_v3, one factor changed at a time (the pinned run is the baseline):
#   0 r11off      the R11 stream CB correction off (SINGLET_RG_NO_R11=1, experiment binary, not the pin)
#   1 rules0      the #80 intron rules off (SINGLET_INTRON_RULES=0, the pinned binary), R11 on
#   2 both_off    both off (experiment binary)
# Each run is extracted and scored with the counts-family code on hgmm1k_v3 only; the metric is printed at the end.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/release-gate/jobs/rg_common.sh
ID=hgmm1k_v3; SPECIES=GRCh38-GRCm39-barnyard; T=8
NAMES=(r11off rules0 both_off); V=${NAMES[$SLURM_ARRAY_TASK_ID]}
EXP=$RG/exp/$V; rm -rf $EXP; mkdir -p $EXP/pb/runs/$ID $EXP/art/gex-counts
BIN=$RG/bin/$PIN8/singlet-cpu
case $V in r11off|both_off) BIN=$RG/bin/$PIN8-r11off/singlet-cpu-r11off; export SINGLET_RG_NO_R11=1 ;; esac
case $V in rules0|both_off) export SINGLET_INTRON_RULES=0 ;; esac
O=$EXP/pb/runs/$ID/singlet-public; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "[r11] variant=$V bin=$BIN rules=${SINGLET_INTRON_RULES:-default} noR11=${SINGLET_RG_NO_R11:-0}"
D=$PB/data/$ID
cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz; cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
$BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/s.1fq > $O/encode.log 2>&1; rm -f $SCR/R*.fastq.gz
export SINGLET_STALL_TIMEOUT=10800; cd $SCR
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base $REF --genome-dir $REF/$SPECIES/star_2.7.11b --exons $REF/$SPECIES/genes/genes.gtf \
  --out-prefix $O/ --threads $T --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "[r11] run exit $?"
grep -E "stream CB:|\[cb-correct\]" $O/run.log | cut -c1-250
# competitors and data: symlink farm over the real pipeline-benchmark tree
for d in $PB/runs/$ID/*; do n=$(basename $d); [ $n = singlet-public ] || [ ${n#singlet-public} != $n ] || ln -sfn $d $EXP/pb/runs/$ID/$n; done
ln -sfn $PB/data $EXP/pb/data
PY=$W/envs/pbench/bin/python; SRC=$RG/src/counts
$PY -P $SRC/bench/public/capabilities/extract/run_extract.py gex-counts $ID --pb $EXP/pb --out $EXP/art/gex-counts
$PY -P $SRC/bench/public/capabilities/extract/run_extract.py meta $ID --pb $EXP/pb --out $EXP/art/gex-counts || true
cd $SRC && $PY - <<PYEOF
import sys, json
sys.path.insert(0, "bench/public/capabilities"); sys.path.insert(0, "bench/public")
import gex_counts
doc = gex_counts.compute("$EXP/art/gex-counts")
for m in doc["metrics"]:
    if m["id"].startswith("umi_ratio_distance") and m["dataset"] == "$ID":
        print("RESULT $V", m["id"], m["singlet"]["value"], "best competitor", min(c["value"] for c in m["competitors"]))
json.dump({m["id"] + "@" + m["dataset"]: m["singlet"]["value"] for m in doc["metrics"] if m["dataset"] == "$ID"}, open("$EXP/metrics.json", "w"), indent=1)
PYEOF

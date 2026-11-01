#!/bin/bash
#SBATCH --job-name=s8-bisect
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# S8: which commit moved the counts-family metrics between the gate-1 (d9834b8a) and gate-2 (3eaa041c) binaries? One pinned binary per array task (release-gate/bin/<pin>/singlet-cpu),
# dataset ID (default nuclei2k_v2), same inputs and flags as g2_counts_singlet.sbatch; the run is extracted and scored with the gate-2 counts code (gex-counts, spliced-unspliced).
set -euo pipefail
PINS=(d9834b8a f15dad07 72ac6555 c8f76c5c 2c030ed9 03a5fac5 e347b10f 3eaa041c)
export PIN8=${PINS[$SLURM_ARRAY_TASK_ID]}
source /mnt/projects/debruinz_project/debruinz/singlet/release-gate/jobs/rg_common.sh
ID=${ID:-nuclei2k_v2}; T=$SLURM_CPUS_PER_TASK
rg_check_bin
case $ID in hgmm1k_v3) SPECIES=GRCh38-GRCm39-barnyard ;; neuron1k_v3|nuclei2k_v2) SPECIES=GRCm39-2024-A ;; *) SPECIES=GRCh38-2024-A ;; esac
EXP=$RG/bisect/$PIN8/$ID; rm -rf $EXP; mkdir -p $EXP/pb/runs/$ID $EXP/art
O=$EXP/pb/runs/$ID/singlet-public; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
D=$PB/data/$ID
cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz; cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
$BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/s.1fq > $O/encode.log 2>&1; rm -f $SCR/R*.fastq.gz
export SINGLET_STALL_TIMEOUT=10800; cd $SCR
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base $REF --genome-dir $REF/$SPECIES/star_2.7.11b --exons $REF/$SPECIES/genes/genes.gtf \
  --out-prefix $O/ --threads $T --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "[bisect] run exit $?"
rg_stream_ok $O/run.log || true
cp $(dirname $BIN)/commit $O/binary_commit.txt
for d in $PB/runs/$ID/*; do n=$(basename $d); [ $n = singlet-public ] || [ ${n#singlet-public} != $n ] || ln -sfn $d $EXP/pb/runs/$ID/$n; done
ln -sfn $PB/data $EXP/pb/data
PY=$W/envs/pbench/bin/python; SRC=$RG/gate2/src_counts
for CAP in gex-counts spliced-unspliced; do
  mkdir -p $EXP/art/$CAP
  $PY -P $SRC/bench/public/capabilities/extract/run_extract.py $CAP $ID --pb $EXP/pb --out $EXP/art/$CAP
  $PY -P $SRC/bench/public/capabilities/extract/run_extract.py meta $ID --pb $EXP/pb --out $EXP/art/$CAP || true
done
cd $SRC && $PY - <<PYEOF
import sys
sys.path.insert(0, "bench/public/capabilities"); sys.path.insert(0, "bench/public")
import gex_counts, spliced_unspliced
for mod, cap in ((gex_counts, "gex-counts"), (spliced_unspliced, "spliced-unspliced")):
    doc = mod.compute("$EXP/art/" + cap)
    for m in doc["metrics"]:
        if m["id"].startswith("umi_ratio_distance") and m["dataset"] == "$ID":
            print("RESULT", "$PIN8", cap, m["id"], round(m["singlet"]["value"], 5), "best competitor", round(min(c["value"] for c in m["competitors"]), 5))
PYEOF

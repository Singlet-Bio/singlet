#!/bin/bash
#SBATCH --job-name=s8-fixcounts
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# S8: the counts-family metrics (gex-counts, spliced-unspliced) of a build that writes exon_counts / intron_counts as the legacy split and the USA labels as
# spliced_counts / unspliced_counts / ambiguous_counts. BINDIR = a build dir with singlet-gpu (used as the CPU engine) and whitelists beside it.
# The gate-2 extractor reads U = intron_counts and S = exon_counts - ambiguous_counts; here a scratch copy is patched to read S, U, A from the USA matrices
# (exactly the change the gate extractor needs), and the total stays exon_counts + intron_counts.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; RG=$W/release-gate; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:?}; BIN=$BINDIR/singlet-gpu
[ -d $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
DSS=(pbmc1k_v2 nuclei2k_v2 neuron1k_v3 pbmc1k_cite_v3 pbmc1k_v3); ID=${DSS[$SLURM_ARRAY_TASK_ID]}; T=$SLURM_CPUS_PER_TASK
case $ID in neuron1k_v3|nuclei2k_v2) SPECIES=GRCm39-2024-A ;; *) SPECIES=GRCh38-2024-A ;; esac
TAG=$(cut -c1-8 $BINDIR/commit); EXP=$RG/fixcounts/$TAG/$ID; rm -rf $EXP; mkdir -p $EXP/pb/runs/$ID $EXP/art
O=$EXP/pb/runs/$ID/singlet-public; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
D=$PB/data/$ID
cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz; cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
$BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/s.1fq > $O/encode.log 2>&1; rm -f $SCR/R*.fastq.gz
export SINGLET_STALL_TIMEOUT=10800 SINGLET_WHITELIST_DIR=$PB/whitelists; cd $SCR
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base $REF --genome-dir $REF/$SPECIES/star_2.7.11b --exons $REF/$SPECIES/genes/genes.gtf \
  --out-prefix $O/ --threads $T --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "[fixcounts] run exit $?"
grep -q "BAM: unsorted STAR stream into the pileup" $O/run.log && echo "STREAM PATH OK" || echo "STREAM PATH FAIL"
ls $O | grep -E "^(exon|intron|spliced|unspliced|ambiguous)_counts" | tr '\n' ' '; echo
for d in $PB/runs/$ID/*; do n=$(basename $d); [ $n = singlet-public ] || [ ${n#singlet-public} != $n ] || ln -sfn $d $EXP/pb/runs/$ID/$n; done
ln -sfn $PB/data $EXP/pb/data
PY=$W/envs/pbench/bin/python; SRC=$SCR/src_counts; cp -r $RG/gate2/src_counts $SRC
$PY -P - <<PYEOF
import re
p = "$SRC/bench/public/capabilities/extract/runio.py"; s = open(p).read()
s = s.replace('for key, prefix in (("E", "exon_counts"), ("U", "intron_counts"), ("A", "ambiguous_counts")):',
              'for key, prefix in (("E", "exon_counts"), ("U", "intron_counts"), ("A", "ambiguous_counts"), ("Su", "spliced_counts"), ("Uu", "unspliced_counts")):')
open(p, "w").write(s)
p = "$SRC/bench/public/capabilities/extract/counts.py"; s = open(p).read()
a = s.index("    E, U, A = (s[k].pad(universe)")
b = s.index("    S.X.eliminate_zeros()") + len("    S.X.eliminate_zeros()")
s = s[:a] + '    S, U, A = (s[k].pad(universe) for k in ("Su", "Uu", "A"))   # the USA matrices' + s[b:]
open(p, "w").write(s)
PYEOF
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
            print("RESULT", "$TAG", cap, m["id"], round(m["singlet"]["value"], 5), "best competitor", round(min(c["value"] for c in m["competitors"]), 5))
PYEOF

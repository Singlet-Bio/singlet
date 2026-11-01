#!/bin/bash
#SBATCH --job-name=e19-encode
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --nice=10000
#SBATCH --array=0-2
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# E19 (docs/gpu-engine/fable/E19-release.md): the .1fq of the benchmark samples that have only FASTQs (pipeline-benchmark/data/<s>/ has no
# <s>.1fq), encoded the way pbmc1k_v3's was (pipeline-benchmark/data/pbmc1k_v3/encode.time.txt: singlet-fixed encode --reads R1 R2, lanes
# concatenated in gex_fastqs.txt order) and the release gate's counts_singlet_gpu.sbatch (GEX FASTQs only). Written under
# $W/jobs/fable/release/data/<name>/<name>.1fq (not the shared pipeline-benchmark directory). Array: 0 pbmc1k_v2, 1 pbmc1k_cite_v3 (GEX),
# 2 pbmc1k_cite_v3_adt (the antibody library, for the feature-barcode check). Built on short-gpu without a GPU request.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=${BIN:-$PB/bin/singlet-fixed}
NAMES=(pbmc1k_v2 pbmc1k_cite_v3 pbmc1k_cite_v3_adt); n=${NAMES[$SLURM_ARRAY_TASK_ID]}
case $n in
  pbmc1k_v2|pbmc1k_cite_v3) L=$PB/data/$n/gex_fastqs.txt ;;
  pbmc1k_cite_v3_adt) L=$PB/data/pbmc1k_cite_v3/adt_fastqs.txt ;;
esac
SCR=/tmp/$USER-e19enc-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
for f in $(grep "_R1_" $L); do cat $f >> $SCR/R1.fastq.gz; done
for f in $(grep "_R2_" $L); do cat $f >> $SCR/R2.fastq.gz; done
O=$W/jobs/fable/release/data/$n; mkdir -p $O
echo "$n: $(grep -c _R1_ $L) R1 + $(grep -c _R2_ $L) R2 files from $L; bin $BIN ($(sha256sum $BIN | cut -c1-16)) node $(hostname) $(date -Is)" | tee $O/encode.txt
/usr/bin/time -v -o $O/encode.time.txt $BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/$n.1fq > $O/encode.log 2>&1
mv $SCR/$n.1fq $O/$n.1fq.new && mv $O/$n.1fq.new $O/$n.1fq
ls -l $O | tee -a $O/encode.txt
tr '\r' '\n' < $O/encode.log | grep -a -v "reads (" | tail -12 | cut -c1-200 | tee -a $O/encode.txt
grep -E "Elapsed|Maximum resident" $O/encode.time.txt | tee -a $O/encode.txt

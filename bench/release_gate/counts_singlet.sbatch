#!/bin/bash
#SBATCH --job-name=rg-counts
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --array=0-7%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Release gate: the counts family (gex-counts, spliced-unspliced, junctions-psi, cell-calling) with the pinned binary.
# Same inputs/flags as bench/public/capabilities/clipper/singlet_public.sbatch (bench/cellcall-independent-truth). CPU run replaces
# $PB/runs/<ds>/singlet-public (the 02aa9b9 run is moved to singlet-public-02aa9b9, never deleted); the GPU run goes to
# $RG/gpu/<ds> and is diffed against the CPU matrix. Usage: sbatch --array=0-7%4 [--export=ALL,ENGINE=gpu --partition=gpu ...]
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/release-gate/jobs/rg_common.sh
rg_check_bin
DSS=(pbmc1k_v3 pbmc1k_v2 neuron1k_v3 hgmm1k_v3 pbmc1k_cite_v3 nuclei2k_v2 pbmc5p_v1 kang2018_ctrl)
ID=${DSS[$SLURM_ARRAY_TASK_ID]}; T=$SLURM_CPUS_PER_TASK
case $ID in
  hgmm1k_v3) SPECIES=GRCh38-GRCm39-barnyard ;;
  neuron1k_v3|nuclei2k_v2) SPECIES=GRCm39-2024-A ;;
  *) SPECIES=GRCh38-2024-A ;;
esac
SCR=/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
if [ $ENGINE = gpu ]; then O=$RG/gpu/$ID; else
  O=$PB/runs/$ID/singlet-public
  [ -e $PB/runs/$ID/singlet-public-02aa9b9 ] || { [ -d $O ] && mv $O $PB/runs/$ID/singlet-public-02aa9b9; }
fi
rm -rf $O; mkdir -p $O
echo "[rg-counts] $ID species=$SPECIES engine=$ENGINE node=$(hostname) $(date -Is)"
echo "node: $(hostname); cpu: $(lscpu | sed -n 's/^Model name: *//p')" > $O/node.txt
if [ $ID = kang2018_ctrl ]; then
  R1=$W/modality-bench/genotype_pool/fastq/R1.fastq.gz; R2=$W/modality-bench/genotype_pool/fastq/R2.fastq.gz
else
  D=$PB/data/$ID
  cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz
  cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
  R1=$SCR/R1.fastq.gz; R2=$SCR/R2.fastq.gz
fi
/usr/bin/time -v -o $O/encode.time.txt $BIN encode --reads $R1 $R2 -o $SCR/s.1fq > $O/encode.log 2>&1
[ $ID = kang2018_ctrl ] || rm -f $SCR/R1.fastq.gz $SCR/R2.fastq.gz
export SINGLET_STALL_TIMEOUT=10800
CMD=($BIN $SCR/s.1fq --ref-base $REF --genome-dir $REF/$SPECIES/star_2.7.11b --exons $REF/$SPECIES/genes/genes.gtf
     --out-prefix $O/ --threads $T --pipeline --output-format mtx --raw-matrix "${ENGINE_ARGS[@]}")
echo "${CMD[*]}" > $O/command.txt; cp $(dirname $BIN)/commit $O/binary_commit.txt; sha256sum $BIN > $O/binary_sha256.txt
cd $SCR
/usr/bin/time -v -o $O/time.txt "${CMD[@]}" > $O/run.log 2>&1 || echo "[rg-counts] exit $?"
rg_stream_ok $O/run.log | tee $O/stream_path.txt
grep -E "^\[singlet\] \.1fq:|cell_calling\] called_v2|Total:" $O/run.log | cut -c1-300
grep -E "Elapsed \(wall|Maximum resident" $O/time.txt $O/encode.time.txt

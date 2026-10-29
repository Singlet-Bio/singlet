#!/bin/bash
#SBATCH --job-name=ar-gpu
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --nice=10000
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Run bench/atac_reducers/atac_reducers_test (built by build_test.sbatch) on one H100 NVL:
#   A  real alignment records of chr21/chr22/chrM of the 5k ATAC BAM (prep_a.sbatch) vs the numba reference and the CPU engine's rows
#   B1 all 102.5 M unique fragments of the 5k sample (the CPU engine's fragments.tsv.gz, expanded to its 207.7 M read pairs + decoys) vs the CPU's
#      fragments.tsv, peaks.bed, atac_peak_counts.mtx and per-cell counters, byte for byte
#   B2 the same with the A8 second pass (peaks from the called cells)
# Single GPU, node-local scratch, everything cleaned on exit.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark; export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
O=$PB/runs/atacsp/atac_pbmc_5k_nextgem; G=$PB/runs/atacsp/atac_pbmc_5k_nextgem_gpu; A=$G/modeA; WL=$PB/whitelists/atacsp/737K-cratac-v1.txt
BIN=$PB/bin/atac_reducers_test; BARE=$W/repos/singlet-py.git; BR=${BR:-claude/gpu-atac-reducers-270cf8}; URL=$(git -C $BARE config remote.origin.url)
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT INT TERM
git -C $BARE fetch -q "$URL" +refs/heads/$BR:refs/w11/ar-run && git -C $BARE archive refs/w11/ar-run bench/atac_reducers | tar -x -C $SCR/src
echo "commit $(git -C $BARE rev-parse --short refs/w11/ar-run)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# the binary was built in the CUDA image; it needs only the driver, so run it directly (libcudart is linked statically by nvcc)
ONLY=${ONLY:-A,B1,B2}
if [[ ",$ONLY," == *",A,"* ]]; then
  echo "================ A: real alignment records, chr21 chr22 chrM"; mkdir -p $SCR/A
  /usr/bin/time -v $BIN --chroms $A/chroms.tsv --whitelist $WL --recs $A/recs.bin --cells-bc $A/cells.txt --configs 1:0,1:11,2:11,7:11 --out $SCR/A 2> $SCR/A.time
  grep -E "Elapsed|Maximum resident" $SCR/A.time
  python $SCR/src/bench/atac_reducers/compare_a.py $A $SCR/A $WL
fi
if [[ ",$ONLY," == *",B1,"* ]]; then
  echo "================ B1: all fragments of the 5k sample vs the CPU engine's outputs"; mkdir -p $SCR/B1
  zcat $O/atac_peak_counts_barcodes.tsv.gz > $SCR/cols.txt; awk -F'\t' '{print $1}' $A/chroms.tsv > /dev/null
  /usr/bin/time -v $BIN --chroms $A/chroms.tsv --whitelist $WL --tsv "zcat $O/fragments.tsv.gz" --configs 1:0,1:5,2:5 --out $SCR/B1 --text-frag --peaks-bed --cols $SCR/cols.txt --metrics-tsv 2> $SCR/B1.time
  grep -E "Elapsed|Maximum resident" $SCR/B1.time
  echo "-- cmp fragments.tsv";    cmp <(zcat $O/fragments.tsv.gz) $SCR/B1/frags.tsv && echo "fragments.tsv IDENTICAL ($(wc -l < $SCR/B1/frags.tsv) lines)"
  echo "-- cmp peaks.bed";        cmp $O/peaks.bed $SCR/B1/peaks.bed && echo "peaks.bed IDENTICAL ($(wc -l < $SCR/B1/peaks.bed) peaks)"
  echo "-- cmp atac_peak_counts"; cmp <(zcat $O/atac_peak_counts.mtx.gz) $SCR/B1/atac_peak_counts.mtx && echo "atac_peak_counts.mtx IDENTICAL ($(sed -n 2p $SCR/B1/atac_peak_counts.mtx))"
  echo "-- cmp per-cell counters"; cmp <(awk -F'\t' 'NR>1{print $1"\t"$3"\t"$4"\t"$6"\t"$10"\t"$11}' $O/atac_barcode_metrics.tsv) $SCR/B1/metrics.tsv && echo "per-cell counters IDENTICAL ($(wc -l < $SCR/B1/metrics.tsv) barcodes)"
  echo "-- first differences, if any"
  diff <(zcat $O/fragments.tsv.gz) $SCR/B1/frags.tsv | head -5; diff $O/peaks.bed $SCR/B1/peaks.bed | head -6
fi
if [[ ",$ONLY," == *",B2,"* ]]; then
  echo "================ B2: A8 second pass (peaks from the called cells) on the 5k sample"; mkdir -p $SCR/B2
  /usr/bin/time -v $BIN --chroms $A/chroms.tsv --whitelist $WL --tsv "zcat $O/fragments.tsv.gz" --cells-bc $O/atac_cells.tsv --configs 1:0,2:5 --out $SCR/B2 --peaks-bed 2> $SCR/B2.time
  grep -E "Elapsed|Maximum resident" $SCR/B2.time
  echo "CPU peaks (all barcodes): $(wc -l < $O/peaks.bed); A8 peaks: $(wc -l < $SCR/B2/peaks.bed)"
  mkdir -p $G; cp $SCR/B2/peaks.bed $G/peaks_a8.bed
fi
echo "[done]"

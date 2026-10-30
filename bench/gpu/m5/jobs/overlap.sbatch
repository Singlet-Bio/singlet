#!/bin/bash
#SBATCH --job-name=w11-ovl
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Cross-sample reuse of mapping results (exploration.md §12.16): a sequence's mapping result depends on the sequence only, so a
# result computed for one sample serves every later sample with the same cDNA sequence. For pairs of real samples: the share of
# sample B's reads (and distinct sequences) whose exact cDNA sequence occurs in sample A. Reads stay in job /tmp.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; DEC=$PB/bin/singlet-w3
SCR=/tmp/$USER-ovl-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; export LC_ALL=C
declare -A F=( [pbmc1k_v3]=$PB/data/pbmc1k_v3/pbmc1k_v3.1fq [GSM7656149]=$W/data/atlas-rerun/perf/GSM7656149/sample.1fq
               [GSM7036403]=$W/data/atlas-rerun/perf/GSM7036403/sample.1fq [pbmc5p_v1]=$PB/data/pbmc5p_v1/pbmc5p_v1.1fq
               [GSM6470208]=$W/data/atlas-rerun/perf/GSM6470208/sample.1fq )
for s in "${!F[@]}"; do
  [ -e "${F[$s]}" ] || { echo "missing $s"; continue; }
  $DEC decode "${F[$s]}" --r2 /dev/stdout --threads 4 2>/dev/null | awk 'NR%4==2' | sort -S 30G --parallel=8 -T $SCR | uniq -c \
    | awk '{print $2"\t"$1}' > $SCR/$s.tsv &
done; wait
for s in "${!F[@]}"; do [ -s $SCR/$s.tsv ] && echo "$s distinct=$(wc -l < $SCR/$s.tsv) reads=$(awk -F'\t' '{s+=$2}END{print s}' $SCR/$s.tsv) len=$(head -1000 $SCR/$s.tsv | awk '{print length($1)}' | sort -n | uniq -c | sort -k1,1nr | head -1)"; done
pair() {   # share of B covered by A
  join -t $'\t' -j 1 -o 2.2 $SCR/$1.tsv $SCR/$2.tsv | awk -v a=$1 -v b=$2 -v nb=$(wc -l < $SCR/$2.tsv) -v rb=$(awk -F'\t' '{s+=$2}END{print s}' $SCR/$2.tsv) \
    '{n++; r+=$1} END {printf "A=%s B=%s shared_distinct=%d (%.4f of B) shared_reads_of_B=%d (%.4f)\n", a, b, n, n/nb, r, r/rb}'
}
pair GSM7656149 pbmc1k_v3; pair pbmc1k_v3 GSM7656149; pair GSM7036403 pbmc1k_v3; pair GSM6470208 pbmc5p_v1; pair pbmc5p_v1 GSM6470208
# A = union of the other 3' human samples
sort -m -t $'\t' -k1,1 $SCR/GSM7656149.tsv $SCR/GSM7036403.tsv | awk -F'\t' '$1!=p{print; p=$1}' > $SCR/u.tsv
pair u pbmc1k_v3

#!/bin/bash
#SBATCH --job-name=ar-prepA
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Mode A inputs: real alignment records of chr21, chr22 and chrM of the kept 5k ATAC BAM (one streaming pass, no index exists), barcodes resolved by exact
# whitelist hit (layout offset 0 forward), plus the numba reference outputs from the same records. Writes $W/pipeline-benchmark/runs/atacsp/atac_pbmc_5k_nextgem_gpu/modeA/.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
PB=$W/pipeline-benchmark; BAM=${BAM:-$PB/runs/atacsp/atac_pbmc_5k_nextgem/atac_sorted.bam}; WL=$PB/whitelists/atacsp/737K-cratac-v1.txt
CONTIGS=${CONTIGS:-chr21,chr22,chrM}; OUT=${OUT:-$PB/runs/atacsp/atac_pbmc_5k_nextgem_gpu/modeA}; mkdir -p $OUT
BARE=$W/repos/singlet-py.git; BR=${BR:-claude/gpu-atac-reducers-270cf8}; URL=$(git -C $BARE config remote.origin.url)
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT INT TERM
git -C $BARE fetch -q "$URL" +refs/heads/$BR:refs/w11/ar-prep && git -C $BARE archive refs/w11/ar-prep bench/atac_reducers | tar -x -C $SCR/src
samtools view -H $BAM | awk -F'\t' '/^@SQ/{sub("SN:","",$2); sub("LN:","",$3); print $2"\t"$3}' > $OUT/chroms.tsv
EXPR=$(echo $CONTIGS | awk -F, '{s=""; for(i=1;i<=NF;i++){s=s (i>1?"||":"") "rname==\"" $i "\""}; print s}')
echo "[prep] filter: $EXPR"; date +%T
samtools view -@ 6 -e "$EXPR" $BAM | awk -F'\t' -v OFS=' ' 'FILENAME==ARGV[1]{wl[$1]=FNR-1; next} FILENAME==ARGV[2]{tid[$1]=FNR-1; next}
  { bc=substr($1,4,16); b=(bc in wl)?wl[bc]:-1; mt=($7=="=")?tid[$3]:(($7 in tid)?tid[$7]:-1); print b, tid[$3], $4-1, $9, mt, $2, $5 }' $WL $OUT/chroms.tsv - > $SCR/recs.txt
echo "[prep] records $(wc -l < $SCR/recs.txt)"; date +%T
awk -v C=",$CONTIGS," 'index(C, "," $1 ",")' <(zcat $(dirname $BAM)/fragments.tsv.gz) > $OUT/cpu_frags_sub.tsv; echo "[prep] cpu fragments in these contigs: $(wc -l < $OUT/cpu_frags_sub.tsv)"
python -u $SCR/src/bench/atac_reducers/prep_a.py $SCR/recs.txt $OUT/chroms.tsv $WL $OUT
echo "[prep] done"; ls -la $OUT

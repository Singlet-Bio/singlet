#!/bin/bash
#SBATCH --job-name=rg-det
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=03:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# alignment-performance / thread_count_outputs_identical on the pinned binary: pbmc1k_v3, one .1fq, 4 / 8 / 16 threads, data outputs
# compared by the SHA-256 of their decompressed bytes (the 10-26 gate had 0: outputs differed with the thread count).
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/release-gate/jobs/rg_common.sh; rg_check_bin
ID=pbmc1k_v3; SPECIES=GRCh38-2024-A; D=$PB/data/$ID; OUT=$RG/det; rm -rf $OUT; mkdir -p $OUT
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz; cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
$BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/s.1fq > $OUT/encode.log 2>&1; rm -f $SCR/R*.fastq.gz
export SINGLET_STALL_TIMEOUT=10800; cd $SCR
for T in 4 8 16; do
  O=$OUT/t$T; mkdir -p $O
  /usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base $REF --genome-dir $REF/$SPECIES/star_2.7.11b --exons $REF/$SPECIES/genes/genes.gtf \
    --out-prefix $O/ --threads $T --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "[det] T=$T exit $?"
  rg_stream_ok $O/run.log || true
  ( cd $O; for f in $(find . -maxdepth 1 -type f -printf '%f
' | sort | grep -Ev "\.log$|\.txt$|time|\.bam|\.fq|STARtmp|_Log|\.progress|read_stats|\.json$|^node|^command"); do
      case $f in *.gz) h=$(zcat $f | sha256sum | cut -c1-16);; *) h=$(sha256sum $f | cut -c1-16);; esac; echo "$f $h"; done ) > $OUT/hash_t$T.txt
done
echo "== files differing between thread counts (4 vs 8, 4 vs 16):"
diff $OUT/hash_t4.txt $OUT/hash_t8.txt || true; diff $OUT/hash_t4.txt $OUT/hash_t16.txt || true
echo "n files $(wc -l < $OUT/hash_t4.txt); identical 4v8: $(comm -12 <(sort $OUT/hash_t4.txt) <(sort $OUT/hash_t8.txt) | wc -l); 4v16: $(comm -12 <(sort $OUT/hash_t4.txt) <(sort $OUT/hash_t16.txt) | wc -l)"
grep -E "Elapsed \(wall|Maximum resident" $OUT/t*/time.txt

#!/bin/bash
#SBATCH --job-name=cnt-sweep
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=03:55:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Count-rule sweep on development libraries (CPU only). Runs the engine with SINGLET_INTRON_RULES=<value> and scores gex-counts and
# spliced-unspliced against the stored comparators (same layout and scorer as jobs/d100score/nucsweep.sbatch).
# Needs: BIN (path of a singlet-gpu binary, run without --engine gpu), OUT (results root), JOBS (file with lines "<rules> <library>"),
#        optional EXTRA_ENV ("K=V K=V") exported to every run.
set -uo pipefail
: "${BIN:?}" "${OUT:?}" "${JOBS:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
export BIN OUT
run1() {
  T=$1; ID=$2; W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
  J=$OUT; export SINGLET_INTRON_RULES=$T; for kv in ${EXTRA_ENV:-}; do export "$kv"; done
  case $ID in hgmm1k_v3) SPECIES=GRCh38-GRCm39-barnyard ;; neuron1k_v3|nuclei2k_v2) SPECIES=GRCm39-2024-A ;; *) SPECIES=GRCh38-2024-A ;; esac
  O=$J/r$T/pb/runs/$ID/singlet-public
  SCR=/tmp/$USER-cnt-$T-$ID; rm -rf $SCR; mkdir -p $SCR; export TMPDIR=$SCR SINGLET_STALL_TIMEOUT=10800
  mkdir -p $(dirname $O); rm -rf $O; mkdir -p $O
  for d in $PB/runs/$ID/*; do n=$(basename $d); case $n in singlet-public*) ;; *) ln -sfn $d $J/r$T/pb/runs/$ID/$n ;; esac; done; ln -sfn $PB/data $J/r$T/pb/data
  D=$PB/data/$ID; cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz; cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
  echo "[$T $ID] start $(date -Is)"
  $BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/s.1fq > $O/encode.log 2>&1 || { echo "[$T $ID] encode failed"; tail -3 $O/encode.log; return; }
  rm -f $SCR/R?.fastq.gz; cd $SCR; G=$REF/$SPECIES
  /usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $O/ --threads 8 --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1
  echo "[$T $ID] exit $? $(date -Is)"; rm -rf $SCR
}
export -f run1
xargs -P 3 -L 1 bash -c 'run1 "$@"' _ < $JOBS
RG=$W/release-gate4; PY=$W/envs/pbench/bin/python; SRC=$RG/gate2/src_counts
for T in $(awk '{print $1}' $JOBS | sort -u); do F=$OUT/r$T/pb; ART=$OUT/r$T/artifacts; RES=$OUT/r$T/results; mkdir -p $ART $RES
  IDS=$(awk -v t=$T '$1==t{print $2}' $JOBS)
  for CAP in gex-counts spliced-unspliced; do for ID in $IDS; do mkdir -p $ART/$CAP; O=$F/runs/$ID/singlet-public; echo sweep > $O/binary_commit.txt; sha256sum $BIN > $O/binary_sha256.txt
      $PY -P $SRC/bench/public/capabilities/extract/run_extract.py $CAP $ID --pb $F --out $ART/$CAP > /dev/null 2>&1; $PY -P $SRC/bench/public/capabilities/extract/run_extract.py meta $ID --pb $F --out $ART/$CAP > /dev/null 2>&1; done
    (cd $SRC && $PY bench/public/run_capability.py $CAP $ART/$CAP --out $RES/$CAP.json > $OUT/r$T/cap_$CAP.log 2>&1); done
done
echo "all done $(date -Is)"

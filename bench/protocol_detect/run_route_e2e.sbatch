#!/bin/bash
#SBATCH --job-name=detect-e2e
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Routing exit test: one library per non-10x assay goes through the engine twice from the same reads, in auto mode and with the
# protocol forced, and the two runs are compared. Needs BIN, SRC and an array over the lines of E2E_TSV (id, r1_url, r2_url, forced tag,
# organism, reads). Each run: `singlet download --r1 --r2` (auto, or --protocol TAG), then the CPU pipeline (--output-format mtx).
# Compared: the decoded read pairs, the .1fq header minus its provenance fields, and the count matrix files. Output: $W/jobs/detect/e2e-<job>/<id>/.
set -uo pipefail
: "${BIN:?}" "${E2E_TSV:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference; T=${SLURM_CPUS_PER_TASK:-16}
export SINGLET_WHITELIST_DIR=${WLDIR:-$W/pipeline-benchmark/whitelists} SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
IFS=$'\t' read -r id r1 r2 tag org nreads < <(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" $E2E_TSV)
O=$W/jobs/detect/e2e-${SLURM_ARRAY_JOB_ID}/$id; mkdir -p $O; SCR=/tmp/$USER-e2e-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
case $org in *Mus*) G=$REF/GRCm39-2024-A;; *) G=$REF/GRCh38-2024-A;; esac
echo "$id $tag $org reads=$nreads genome=$G bin=$BIN" | tee $O/summary.txt
for k in 1 2; do u=$([ $k = 1 ] && echo $r1 || echo $r2)
  ( curl -sS --retry 3 "$u" 2>/dev/null | gzip -dc 2>/dev/null | head -$((4 * nreads)) > $SCR/f$k.fastq ) &
done; wait
echo "pairs: $(( $(wc -l < $SCR/f1.fastq) / 4 ))" | tee -a $O/summary.txt
for mode in auto forced; do
  extra=; [ $mode = forced ] && extra="--protocol $tag"
  $BIN download --r1 $SCR/f1.fastq --r2 $SCR/f2.fastq --staging-dir $SCR/st-$mode -o $SCR/$mode.1fq $extra > $O/$mode.download.log 2>&1; echo "$mode download exit $?" | tee -a $O/summary.txt
  grep -a -E "\[singlet-gate\]|Protocol:" $O/$mode.download.log | cut -c1-260 | tee -a $O/summary.txt
  $BIN decode $SCR/$mode.1fq --header 2>&1 | grep -a -v -E "protocol_source|protocol_confidence|protocol_detector|protocol_geo|protocol_evidence|source_label|created|timestamp|version|commit|input|path|elapsed" > $O/$mode.header.txt
  $BIN decode $SCR/$mode.1fq --r1 $SCR/$mode.r1.fq --r2 $SCR/$mode.r2.fq --threads 8 > /dev/null 2>&1
  echo "$mode reads md5 $(cat $SCR/$mode.r1.fq $SCR/$mode.r2.fq | md5sum | cut -c1-16)" | tee -a $O/summary.txt; rm -f $SCR/$mode.r1.fq $SCR/$mode.r2.fq
  mkdir -p $SCR/run-$mode/out
  ( cd $SCR/run-$mode && $BIN $SCR/$mode.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/run-$mode/out/ \
      --threads $T --pipeline --output-format mtx > $O/$mode.pipe.log 2>&1 ); echo "$mode pipeline exit $?" | tee -a $O/summary.txt
  cp $SCR/run-$mode/out/summary.json $O/$mode.summary.json 2>/dev/null
  ( cd $SCR/run-$mode/out && find . -type f \( -name "*.mtx*" -o -name "barcodes*" -o -name "features*" -o -name "genes*" \) | sort | xargs -r sha256sum | cut -c1-16,65- ) > $O/$mode.matrix.sha 2>/dev/null
  echo "$mode matrix files: $(wc -l < $O/$mode.matrix.sha)" | tee -a $O/summary.txt
done
cmp -s $O/auto.header.txt $O/forced.header.txt && h=same || h=DIFF
cmp -s $O/auto.matrix.sha $O/forced.matrix.sha && [ -s $O/auto.matrix.sha ] && m=same || m=DIFF
echo "RESULT $id header=$h matrix=$m" | tee -a $O/summary.txt
diff $O/auto.header.txt $O/forced.header.txt | head -20 | tee -a $O/summary.txt
echo done $(date -Is)

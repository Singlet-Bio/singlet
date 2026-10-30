#!/bin/bash
#SBATCH --job-name=if-sweep
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --array=0-6
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# bench/public singlet_public.sbatch (encode from the FASTQs, campaign flags, own reference) with bin/$BIN, plus
#   SINGLET_INTRON_SWEEP=$SWEEP  -> runs/<ds>/$TAG/intron_sweep/{main,rules_<r>}.tsv.gz
#   SINGLET_MOLECULE_DUMP        -> runs/<ds>/$TAG/molecules.tsv.gz (main pass; DUMP=1)
#   SINGLET_INTRON_RULES=$RULES  -> rules of the main pass (default: the binary's)
# Env: BIN (default singlet-intronfix), TAG (default singlet-if-sweep).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference
ALL=(pbmc1k_v3 nuclei2k_v2 hgmm1k_v3 pbmc1k_cite_v3 pbmc1k_v2 neuron1k_v3 pbmc5p_v1)
ID=${ALL[$SLURM_ARRAY_TASK_ID]}; T=$SLURM_CPUS_PER_TASK
BIN=$PB/bin/${BIN:-singlet-intronfix}; TAG=${TAG:-singlet-if-sweep}
case $ID in
  hgmm1k_v3) SPECIES=GRCh38-GRCm39-barnyard ;;
  neuron1k_v3|nuclei2k_v2) SPECIES=GRCm39-2024-A ;;
  *) SPECIES=GRCh38-2024-A ;;
esac
O=$PB/runs/$ID/$TAG; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "$ID node=$(hostname) bin=$(sha256sum $BIN | cut -c1-16) sweep=${SWEEP:-} rules=${RULES:-} dump=${DUMP:-}" | tee $O/node.txt
D=$PB/data/$ID
cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz
cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
$BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/s.1fq > $O/encode.log 2>&1
rm -f $SCR/R1.fastq.gz $SCR/R2.fastq.gz
export SINGLET_STALL_TIMEOUT=10800
[ -n "${SWEEP:-}" ] && export SINGLET_INTRON_SWEEP=$SWEEP
[ -n "${RULES:-}" ] && export SINGLET_INTRON_RULES=$RULES
[ -n "${DUMP:-}" ] && export SINGLET_MOLECULE_DUMP=$O/molecules.tsv.gz
CMD=($BIN $SCR/s.1fq --ref-base $REF --genome-dir $REF/$SPECIES/star_2.7.11b --exons $REF/$SPECIES/genes/genes.gtf
     --out-prefix $O/ --threads $T --pipeline --output-format mtx --raw-matrix)
echo "${CMD[*]}" > $O/command.txt
cd $SCR
/usr/bin/time -v -o $O/time.txt "${CMD[@]}" > $O/run.log 2>&1 || echo "exit $?" | tee -a $O/run.log
grep -E "^\[sweep\]|molecule dump|intron rules" $O/run.log || true
grep -E "Elapsed \(wall|Maximum resident" $O/time.txt
for f in exon_counts.mtx.gz intron_counts.mtx.gz ambiguous_counts.mtx.gz; do
  if cmp -s <(zcat $O/$f | grep -v "^%") <(zcat $PB/runs/$ID/singlet-public/$f | grep -v "^%"); then echo "same as singlet-public: $f"; else echo "DIFFERS from singlet-public: $f"; fi
done

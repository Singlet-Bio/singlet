#!/bin/bash
#SBATCH --job-name=pe1d-index
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=03:50:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Phase 1d: GRCh38 STAR index WITHOUT sjdb, same parameters as the engine's index
# (/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b: genomeSAindexNbases 14, genomeSAsparseD 2, chrBinNbits 18).
# CPU only. Output: $OUT (kept; the only thing from this study that stays on Clipper).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
OUT=${OUT:-$W/data/w11/GRCh38-2024-A-nosjdb}
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR; [ -e $OUT.partial ] && rm -rf $OUT.partial' EXIT
[ -e $OUT/SA ] && { echo "already built: $OUT"; exit 0; }
rm -rf $OUT.partial; mkdir -p $OUT.partial
$ENV/bin/STAR --version
timeout 3h $ENV/bin/STAR --runMode genomeGenerate --runThreadN 24 --genomeDir $OUT.partial \
  --genomeFastaFiles $REF/fasta/genome.fa --genomeSAindexNbases 14 --genomeSAsparseD 2 --genomeChrBinNbits 18 \
  --outTmpDir $SCR/startmp --outFileNamePrefix $SCR/gen_
cp $SCR/gen_Log.out $OUT.partial/generate.Log.out
ls $OUT.partial/sjdb* 2>&1 || echo "no sjdb files (expected)"
ls -la $OUT.partial; cat $OUT.partial/genomeParameters.txt
mv $OUT.partial $OUT; trap 'rm -rf $SCR' EXIT
echo "[index] ok $OUT"

#!/bin/bash
#SBATCH --job-name=edit-panel
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# D-101: download REDIportal v3 (human hg38, mouse mm10) and build the engine's editing panels
#   $REF/pileup_indices/rna_editing_GRCh38.{tsv.gz,json}  and  rna_editing_GRCm39.{tsv.gz,json} (mm10 -> GRCm39 by the UCSC chain)
# with scripts/editing/build_rediportal_panel.py from branch BR. Downloads are kept in $W/data/rediportal/ (source files, sha256 in the
# manifests). OUT (default $REF/pileup_indices) is where the panels go; the engine's --pipeline default reads them there.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference
BR=${BR:-feat/rna-editing}; OUT=${OUT:-$REF/pileup_indices}; D=$W/data/rediportal; mkdir -p $D
BARE=$W/repos/singlet-py.git; URL="$(git -C $BARE config remote.origin.url)"
SCR=/tmp/$USER-ep-$SLURM_JOB_ID; mkdir -p $SCR/src $SCR/py; trap 'rm -rf $SCR' EXIT
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$BR:refs/gd/ep-$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/gd/ep-$SLURM_JOB_ID scripts/editing | tar -x -C $SCR/src
echo "commit $(git -C $BARE rev-parse refs/gd/ep-$SLURM_JOB_ID)"
SRCURL=https://rediportal.cloud.ba.infn.it/download
for f in TABLE1_hg38_v3.txt.gz TABLE1_mm10_v3.txt.gz; do
  [ -s $D/$f ] || curl -fsSk --retry 5 -o $D/$f.part $SRCURL/$f && { [ -s $D/$f ] || mv $D/$f.part $D/$f; }
  ls -l $D/$f
done
[ -s $D/mm10ToMm39.over.chain.gz ] || curl -fsS --retry 5 -o $D/mm10ToMm39.over.chain.gz https://hgdownload.soe.ucsc.edu/goldenPath/mm10/liftOver/mm10ToMm39.over.chain.gz
IR=$W/src/singlet-agents/main/hpc/clipper/in_runtime.sh; [ -f $IR ] || IR=$(ls -d $W/src/singlet-agents/*/hpc/clipper/in_runtime.sh | head -1)
source $IR
in_runtime bash -c "pip install -q --target $SCR/py pyliftover"
B=$SCR/src/scripts/editing/build_rediportal_panel.py
in_runtime bash -c "PYTHONPATH=$SCR/py python3 -I $B --table $D/TABLE1_mm10_v3.txt.gz --fasta $REF/GRCm39-2024-A/fasta/genome.fa \
  --snp-panel $REF/pileup_indices/mus_musculus.vcf.gz --chain $D/mm10ToMm39.over.chain.gz --version 'REDIportal v3 (mouse, mm10)' \
  --source-url $SRCURL/TABLE1_mm10_v3.txt.gz --assembly-source mm10 --assembly-target GRCm39 --out $SCR/rna_editing_GRCm39"
in_runtime bash -c "PYTHONPATH=$SCR/py python3 -I $B --table $D/TABLE1_hg38_v3.txt.gz --fasta $REF/GRCh38-2024-A/fasta/genome.fa \
  --snp-panel $REF/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz --version 'REDIportal v3 (human, hg38)' \
  --source-url $SRCURL/TABLE1_hg38_v3.txt.gz --assembly-source hg38 --assembly-target GRCh38 --out $SCR/rna_editing_GRCh38"
for g in GRCm39 GRCh38; do cp $SCR/rna_editing_$g.json $SCR/rna_editing_$g.tsv.gz $OUT/; done
mkdir -p $W/jobs/editing/results/panel-$SLURM_JOB_ID; cp $SCR/rna_editing_*.json $W/jobs/editing/results/panel-$SLURM_JOB_ID/
ls -l $OUT/rna_editing_*

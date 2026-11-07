#!/bin/bash
# Proof of the published reference tars: fetch.sh downloads them from data.singlet.bio into a clean directory and checks their
# sha256, then the 9-sample gate's pbmc2m runs on the GPU with that tree as --ref-base and is compared with the existing CPU reference.
# The rest of the tree (FASTA, GTF, SNP panel, the sample's .1fq) is linked from Clipper, so fetch.sh downloads only the two tars.
#   sbatch --export=ALL,FETCH=<fetch.sh>,BINDIR=<multi-arch build>,REFD=<CPU ref root> proof.sbatch
#SBATCH --job-name=pub-proof
#SBATCH --account=debruinz
#SBATCH --partition=gpu
#SBATCH --gres=gpu:quadro_rtx_8000:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=170G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; R=/mnt/projects/debruinz_project/cellarium/reference; PB=$W/pipeline-benchmark
: "${FETCH:?}" "${BINDIR:?}" "${REFD:?}"
SCR=/tmp/$USER-pubproof-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
D=$SCR/data; G=$D/ref/GRCh38-2024-A
mkdir -p $G/fasta $G/genes $D/ref/pileup_indices $D/pbmc1k_v3
ln -s $R/GRCh38-2024-A/fasta/genome.fa $G/fasta/genome.fa; ln -s $R/GRCh38-2024-A/genes/genes.gtf $G/genes/genes.gtf
ln -s $R/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz $D/ref/pileup_indices/
ln -s $PB/data/pbmc1k_v3/pbmc1k_v3.1fq $D/pbmc1k_v3/pbmc1k_v3.1fq
echo "node=$(hostname) job=$SLURM_JOB_ID $(date -Is); clean tree $D"
t0=$(date +%s)
DATA=$D SINGLET_BIN= bash $FETCH download
echo "fetch.sh download: $(( $(date +%s) - t0 )) s"; cat $D/fetch-report.txt
ls -la $G/star_2.7.11b $D/ref/nonhost
# the rest of Clipper's tree, linked (everything the two tars do not replace)
for e in $R/*; do b=$(basename $e); [ -e $D/ref/$b ] || ln -s $e $D/ref/$b; done
for e in $R/GRCh38-2024-A/*; do b=$(basename $e); [ -e $G/$b ] || ln -s $e $G/$b; done
for e in $R/pileup_indices/*; do b=$(basename $e); [ -e $D/ref/pileup_indices/$b ] || ln -s $e $D/ref/pileup_indices/$b; done
echo "## downloaded files against Clipper's"
( cd $G && for f in star_2.7.11b/*; do cmp -s $f $R/GRCh38-2024-A/$f && echo "same $f" || echo "DIFFERS $f"; done )
( cd $D/ref && for f in nonhost/*.snhskidx nonhost/host_21mer.bloom; do cmp -s $f $R/$f && echo "same $f" || echo "DIFFERS $f"; done )
sed "s|^W=/mnt/projects/debruinz_project/debruinz/singlet; PB=\$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference|W=/mnt/projects/debruinz_project/debruinz/singlet; PB=\$W/pipeline-benchmark; REF=$D/ref|" \
  $W/jobs/gpu-pr2/pr2-gate.sbatch > $SCR/gate.sh
grep -q "REF=$D/ref" $SCR/gate.sh || { echo "gate REF not replaced"; exit 2; }
export BINDIR REFD T=18 SAMPLES=pbmc2m MODES="cpu gpu" SINGLET_ENGINE_GPU_MIN_CC=70
bash $SCR/gate.sh

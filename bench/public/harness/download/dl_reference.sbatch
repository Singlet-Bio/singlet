#!/bin/bash
#SBATCH --job-name=bp-dl-ref
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
#   sbatch dl_reference.sbatch
# The source of truth for the download benchmark: each run is fetched once with `prefetch` (the whole .sra, kept for the
# S3 comparison), and decoded with NCBI's own `fasterq-dump --split-files`. Output (per run): spot count and the
# order-independent digests of every decoded file (fq_digest.py) in $BP/download/reference/reference.tsv, and the sha256 and
# size of the .sra in sra.tsv. Every download tool is later scored against these.
set -eu
source ${BP_SRC:-/mnt/projects/debruinz_project/debruinz/singlet/bench-public/src/singlet}/bench/public/harness/perf_common.sh
source $SRC/bench/public/harness/download/runs.sh
export PATH=$ENVDIR/bin:$PATH
R=$BP/download/reference; mkdir -p $R/sra
export HOME=$R/home; mkdir -p $HOME/.ncbi; cp $WORK/cache/condahome/.ncbi/user-settings.mkfg $HOME/.ncbi/
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/fq $SCR/tmp; trap 'rm -rf $SCR' EXIT
echo "[dl-ref] prefetch $(prefetch --version | tail -1) fasterq-dump $(fasterq-dump --version | tail -1)"
: > $R/sra.tsv; : > $R/reference.tsv
for r in $RUNS; do
  [ -s $R/sra/$r/$r.sra ] || prefetch --max-size u -O $R/sra $r > $R/prefetch_$r.log 2>&1
  echo -e "$r\t$(stat -c %s $R/sra/$r/$r.sra)\t$(sha256sum $R/sra/$r/$r.sra | cut -d' ' -f1)" >> $R/sra.tsv
  fasterq-dump --split-files -e 16 -t $SCR/tmp -O $SCR/fq $R/sra/$r/$r.sra > $R/fasterq_$r.log 2>&1
  ls -la $SCR/fq
done
cd $SCR/fq
ls *.fastq | xargs -P 8 -I{} $PY -I $SRC/bench/public/harness/download/fq_digest.py {} | sort >> $R/reference.tsv
cat $R/sra.tsv $R/reference.tsv

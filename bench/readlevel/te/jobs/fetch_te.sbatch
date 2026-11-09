#!/bin/bash
#SBATCH --job-name=te-fetch
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# TE benchmark data (open access):
#   a375_10x  scNanoGPS (Shiau et al. 2023, GSE212945): A375 nuclei, 10x 3' short reads SRR21492169; nanopore reads of the same
#             library SRR34318914 (long-read truth for subfamily abundance)
#   mesc_2c   Iturbide et al. 2021 (E-MTAB-8869, PRJEB37158): mouse E14 ESC with a 2C::tbGFP reporter, retinoic acid 48 h with LIF
#             (most 2C-like cells), 10x 3' v3, ERR3981386 from ENA
# WHAT: any of "a375_10x a375_ont mesc_2c" (default all). Output: $W/data/readlevel/te/<name>/<name>_S1_L001_R{1,2}_001.fastq.gz
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/sratools/bin:$PATH
D=$W/data/readlevel/te; mkdir -p $D; cd $D; TMP=/tmp/$USER-tf-$SLURM_JOB_ID; trap 'rm -rf $TMP' EXIT
for w in ${WHAT:-a375_10x mesc_2c a375_ont}; do
  mkdir -p $w; [ -f $w/DONE ] && continue
  rm -f $w/*.fastq   # a cancelled run leaves partial fasterq output
  case $w in
    a375_10x)
      fasterq-dump --split-files --include-technical -e 8 -t $TMP -O $w SRR21492169 || exit 1
      ls -l $w; for f in $w/*.fastq; do echo "$f $(head -2 $f | tail -1 | awk '{print length($0)}')"; done ;;
    mesc_2c)
      for k in 1 2; do curl -sSfL -C - -o $w/ERR3981386_$k.fastq.gz ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR398/006/ERR3981386/ERR3981386_$k.fastq.gz || exit 1; done ;;
    a375_ont)
      prefetch -O $TMP --max-size 200G SRR34318914 && fasterq-dump -e 8 -t $TMP -O $w $TMP/SRR34318914/SRR34318914.sra || exit 1 ;;
  esac
  for f in $w/*.fastq; do [ -f $f ] && gzip -1 $f; done
  ls -l $w > $w/DONE
done
# 10x names for the engine (lengths printed above decide which file is R1 for a375_10x)
ls -l */

#!/bin/bash
#SBATCH --job-name=identify-species
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Species filters on real runs: human 10x (SRA), mouse Visium (SRA), a human tumor xenograft in mouse (local FASTQ head).   sbatch species_check.sbatch BIN OUTNAME
# SPDIR holds symlinks to every filter file plus species_filters.tsv and active_filters.tsv (reference pack index, singlet-agents ve/refs-pack).
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/identify; ENV=$W/envs/pbench
BIN=$1; OUT=$J/out/$2; mkdir -p $OUT
export LD_LIBRARY_PATH=$ENV/lib PATH=$ENV/bin:$PATH
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/home; trap 'rm -rf $SCR' EXIT; export HOME=$SCR/home
SPDIR=$J/species_dir
X=$W/data/validation-engine-mixed/xeno/GSM7497134
run() {  # id source hint
  printf '%s\t%s\t\n' "$1" "$2" > $SCR/l.tsv; mkdir -p $OUT/$1
  /usr/bin/time -v -o $OUT/$1/time.txt $BIN identify --list $SCR/l.tsv --out-dir $OUT/$1 --whitelist-dir $J/wl --species-filters $SPDIR ${3:+--species-hint "$3"} 2> $OUT/$1/stderr.txt
  grep -e "Elapsed" -e "Maximum resident" $OUT/$1/time.txt | tr '\n' ' '; echo
}
run human_noHint SRR11549020
run human_hint SRR11549020 "Homo sapiens"
run mouse_noHint SRR22816622
run mouse_hintHuman SRR22816622 "Homo sapiens"
run xeno_noHint $X/R1.fastq.gz,$X/R2.fastq.gz
run xeno_hintHuman $X/R1.fastq.gz,$X/R2.fastq.gz "Homo sapiens"
echo done

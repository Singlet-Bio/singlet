#!/bin/bash
#SBATCH --job-name=s8-muscsinglet
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: Singlet (BINDIR's singlet-gpu as the CPU engine) on the MuSC Illumina lane (SRR12282454, 10x v2, 30.3 M read pairs) with the molecule dump on, intron rules RULES (default 389 and 397).
# R1 is cut to the 26 nt of the v2 geometry (16 CB + 10 UMI). Output: $W/s8/musc/singlet_<rules>/ (molecules.tsv.gz, run.log).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCm39-2024-A
RAW=/mnt/projects/debruinz_project/debruinz/ground-truth/raw/MuSC
BINDIR=${BINDIR:?}; BIN=$BINDIR/singlet-gpu; RULES_LIST=${RULES_LIST:-"389 397"}
[ -d $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
echo "commit=$(cat $BINDIR/commit) node=$(hostname) $(date -Is)"
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; export TMPDIR=$SCR
zcat $RAW/SRR12282454_1.fastq.gz | awk 'NR%4==2||NR%4==0{print substr($0,1,26);next}{print}' | gzip -1 > $SCR/R1.fastq.gz
cp $RAW/SRR12282454_2.fastq.gz $SCR/R2.fastq.gz
$BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz --protocol 10x-3p-v2 -o $SCR/x.1fq > $W/s8/musc/encode.log 2>&1 || { tail -20 $W/s8/musc/encode.log; exit 1; }
rm -f $SCR/R*.fastq.gz; tail -3 $W/s8/musc/encode.log
export SINGLET_STALL_TIMEOUT=10800 SINGLET_WHITELIST_DIR=$PB/whitelists
for R in $RULES_LIST; do
  O=$W/s8/musc/singlet_$R; rm -rf $O; mkdir -p $O; cd $SCR
  SINGLET_INTRON_RULES=$R SINGLET_MOLECULE_DUMP=$O/molecules.tsv.gz /usr/bin/time -v -o $O/time.txt $BIN $SCR/x.1fq --ref-base $REF --genome-dir $G/star_2.7.11b \
    --exons $G/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1
  echo "rules=$R exit=$?"; grep -E "Elapsed \(wall|Maximum resident" $O/time.txt; grep -c . $O/run.log
  grep -q "BAM: unsorted STAR stream into the pileup" $O/run.log && echo "STREAM PATH OK" || echo "STREAM PATH FAIL"
  rm -rf $O/private; ls $O
done

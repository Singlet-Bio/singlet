#!/bin/bash
#SBATCH --job-name=mtl-pile
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch pile.sbatch <name> <bam> <cells.txt> [mgatk|maegatk|none]
# Per-cell chrM allele counts from one BAM: mtpile (read and molecule level, strand) and a reference tool on the same
# reads and cells: mgatk bcall (UMI-deduplicated, -ub UB) for the 3' libraries, maegatk bcall (UMI consensus, >= 3
# reads, as the MAESTER authors ran it) for the MAESTER libraries.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; P=$W/envs/pbench; E=$W/envs/mtl; J=$W/jobs/mtl
NAME=$1; BAM=$2; CELLS=$3; TOOL=${4:-none}
O=$W/out/mtdna_lineage/pile/$NAME; mkdir -p $O
if [ ! -x $J/bin/mtpile ] || [ $J/mtpile.cpp -nt $J/bin/mtpile ]; then
  mkdir -p $J/bin
  $P/bin/x86_64-conda-linux-gnu-g++ -std=c++17 -O2 -I$P/include $J/mtpile.cpp -o $J/bin/mtpile -L$P/lib -lhts -Wl,-rpath,$P/lib
fi
cp $CELLS $O/cells.txt
SITES=""; [ -s $O/sites.txt ] && SITES="--sites $O/sites.txt"
/usr/bin/time -v -o $O/mtpile.time.txt $J/bin/mtpile $BAM $O/cells.txt $O/mt --min-mapq 20 --min-bq 10 --threads 8 $SITES
grep -E "Elapsed|Maximum res" $O/mtpile.time.txt; cat $O/mt.stats.json
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
sed 's/-1$//' $O/cells.txt > $SCR/bc.txt
if [ "$TOOL" = mgatk ]; then
  # mgatk expects barcodes exactly as in the CB tag
  samtools_bc=$(timeout 60 $P/bin/samtools view $BAM chrM | head -1000 | grep -o 'CB:Z:[^[:space:]]*' | head -1 || true)
  case "$samtools_bc" in *-1) cp $O/cells.txt $SCR/bc.txt ;; esac
  /usr/bin/time -v -o $O/mgatk.time.txt $E/bin/mgatk bcall -i $BAM -o $O/mgatk -n $NAME -g hg38 -c 16 -bt CB -b $SCR/bc.txt \
    -ub UB -sr -so > $O/mgatk.log 2>&1 || { echo "mgatk failed"; tail -30 $O/mgatk.log; }
  grep -E "Elapsed|Maximum res" $O/mgatk.time.txt || true; ls -la $O/mgatk/final 2>/dev/null || true
elif [ "$TOOL" = maegatk ]; then
  /usr/bin/time -v -o $O/maegatk.time.txt $E/bin/maegatk bcall -i $BAM -o $O/maegatk -n $NAME -g rCRS -c 16 -bt CB -b $SCR/bc.txt \
    -ub UB -mr 3 -sr -so > $O/maegatk.log 2>&1 || { echo "maegatk failed"; tail -30 $O/maegatk.log; }
  grep -E "Elapsed|Maximum res" $O/maegatk.time.txt || true; ls -la $O/maegatk/final 2>/dev/null || true
fi
ls -la $O

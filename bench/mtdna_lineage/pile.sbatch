#!/bin/bash
#SBATCH --job-name=mtl-pile
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch pile.sbatch <name> <bam> <cells.txt | auto:SINGLET_OUT:TRUTH_CELLS[:COLUMN]> [mgatk|maegatk|none]   (env SITES_FILE: 1-based positions)
# Per-cell chrM allele counts from one BAM: mtpile (read and molecule level, strand) and a reference tool on the same
# reads and cells: mgatk bcall (UMI-deduplicated, -ub UB) for the 3' libraries, maegatk bcall (UMI consensus, >= 3
# reads, as the MAESTER authors ran it) for the MAESTER libraries.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; P=$W/envs/pbench; E=$W/envs/mtl; J=$W/jobs/mtl
export PATH=$E/bin:$PATH   # mgatk / maegatk call snakemake, samtools, java from their env
NAME=$1; BAM=$2; CELLS=$3; TOOL=${4:-none}
O=$W/out/mtdna_lineage/pile/$NAME; mkdir -p $O
# private build per job (several pile jobs may run at once)
BIN=/tmp/mtpile-$SLURM_JOB_ID
$P/bin/x86_64-conda-linux-gnu-g++ -std=c++17 -O2 -I$P/include $J/mtpile.cpp -o $BIN -L$P/lib -lhts -Wl,-rpath,$P/lib
case "$CELLS" in
  auto:*) IFS=: read -r _ SOUT TRUTHC COL <<< "$CELLS"; bash $J/mkcells.sh $SOUT $TRUTHC $O/cells.txt $COL ;;
  *) cp $CELLS $O/cells.txt ;;
esac
[ -n "${SITES_FILE:-}" ] && cp $SITES_FILE $O/sites.txt
SITES=""; [ -s $O/sites.txt ] && SITES="--sites $O/sites.txt"
/usr/bin/time -v -o $O/mtpile.time.txt $BIN $BAM $O/cells.txt $O/mt --min-mapq 20 --min-bq 10 --threads 8 $SITES
grep -E "Elapsed|Maximum res" $O/mtpile.time.txt; cat $O/mt.stats.json
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
sed 's/-1$//' $O/cells.txt > $SCR/bc.txt
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn)   # mgatk/maegatk open one BAM per barcode
if [ "$TOOL" = mgatk ]; then
  # mgatk expects barcodes exactly as in the CB tag (Singlet's tagged BAM writes AAAC...-1)
  CB1=$(set +o pipefail; $P/bin/samtools view $BAM chrM:1-3000 | head -200 | grep -o 'CB:Z:[^[:space:]]*' | head -1)
  case "$CB1" in *-1) sed 's/$/-1/' $SCR/bc.txt > $SCR/bc1.txt && mv $SCR/bc1.txt $SCR/bc.txt ;; esac
  echo "CB example $CB1; barcode file $(head -1 $SCR/bc.txt)"
  rm -rf $O/mgatk
  /usr/bin/time -v -o $O/mgatk.time.txt $E/bin/mgatk bcall -i $BAM -o $O/mgatk -n $NAME -g hg38 -c 16 -bt CB -b $SCR/bc.txt \
    -ub UB -sr -so > $O/mgatk.log 2>&1 || { echo "mgatk failed"; tail -30 $O/mgatk.log; }
  grep -E "Elapsed|Maximum res" $O/mgatk.time.txt || true; ls -la $O/mgatk/final 2>/dev/null || true
elif [ "$TOOL" = maegatk ]; then
  rm -rf $O/maegatk
  /usr/bin/time -v -o $O/maegatk.time.txt $E/bin/maegatk bcall -i $BAM -o $O/maegatk -n $NAME -g rCRS -c 16 -bt CB -b $SCR/bc.txt \
    -ub UB -mr 3 -sr -so > $O/maegatk.log 2>&1 || { echo "maegatk failed"; tail -30 $O/maegatk.log; }
  grep -E "Elapsed|Maximum res" $O/maegatk.time.txt || true; ls -la $O/maegatk/final 2>/dev/null || true
fi
ls -la $O

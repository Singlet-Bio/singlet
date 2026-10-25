#!/bin/bash
#SBATCH --job-name=as-chromap
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Open-source FASTQ -> fragments comparator: Chromap (--preset atac) with the 10x whitelist, then MACS3 (Signac recipe) and
# SnapATAC2 QC on its fragments (competitors_atac.py). Same FASTQs, same GRCh38-2024-A, same threads as Singlet.
#   RUN=<name> FQ=<dir> GLOB=<prefix> WL=<whitelist> [BCFMT=bc:0:-1|bc:8:-1|bc:0:-1:-] [TRANSLATE=<atac<TAB>gex file>]
set -u
BARE=/mnt/projects/debruinz_project/debruinz/singlet/repos/singlet-py.git; SRCS=/mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__atac-spatial-benchmark-2ba7df__scripts
git -C $BARE fetch -q --update-head-ok origin +refs/heads/claude/atac-spatial-benchmark-2ba7df:refs/heads/claude/atac-spatial-benchmark-2ba7df; git -C $SRCS checkout -q --detach refs/heads/claude/atac-spatial-benchmark-2ba7df
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/atacbench; PB=$W/pipeline-benchmark
G=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
O=$PB/runs/atacsp/chromap_$RUN; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
R1=$(ls $FQ/${GLOB}*_R1_*.fastq.gz | sort | paste -sd,); R2=$(ls $FQ/${GLOB}*_R2_*.fastq.gz | sort | paste -sd,)
R3=$(ls $FQ/${GLOB}*_R3_*.fastq.gz | sort | paste -sd,)
grep -v '^#' $WL | cut -f1 > $SCR/wl.txt
# Barcode window and strand: the read format with the highest exact whitelist hit rate on 200k reads (the user of
# Chromap has to choose it; Cell Ranger and Singlet detect it).
if [ -z "${BCFMT:-}" ]; then
  BCFMT=$(zcat $(ls $FQ/${GLOB}*_R2_*.fastq.gz | sort | head -1) | head -800000 | awk 'NR%4==2' | $ENV/bin/python -c "
import sys
wl=set(l.strip() for l in open('$SCR/wl.txt'))
rc=lambda s: s[::-1].translate(str.maketrans('ACGTN','TGCAN'))
reads=[l.strip() for l in sys.stdin]
L=len(reads[0]); best=None
for a in sorted({0, L-16}):
    for st in ('', ':-'):
        f=(lambda r,a=a: rc(r[a:a+16])) if st else (lambda r,a=a: r[a:a+16])
        h=sum(f(r) in wl for r in reads)/len(reads)
        print(f'bc:{a}:{a+15}{st}', round(h,4), file=sys.stderr)
        if best is None or h>best[1]: best=(f'bc:{a}:{a+15}{st}', h)
print(best[0])")
fi
echo "barcode read format: $BCFMT"
X=""; [ -n "${TRANSLATE:-}" ] && { zcat -f $TRANSLATE | grep -v '^#' > $SCR/tr.tsv; X="--barcode-translate $SCR/tr.tsv"; }
/usr/bin/time -v -o $O/time.txt $ENV/bin/chromap --preset atac -t 16 -x $W/cache/chromap/GRCh38-2024-A.index -r $G/fasta/genome.fa \
  -1 $R1 -2 $R3 -b $R2 --barcode-whitelist $SCR/wl.txt --read-format r1:0:-1,bc:${BCFMT#bc:},r2:0:-1 $X -o $SCR/fragments.tsv > $O/chromap.log 2>&1
echo "chromap exit=$? wall=$(grep 'Elapsed (wall' $O/time.txt | sed 's/.*: //') rss_kb=$(grep 'Maximum resident' $O/time.txt | sed 's/.*: //')"
tail -15 $O/chromap.log
sort -k1,1 -k2,2n -S 20G -T $SCR $SCR/fragments.tsv | $ENV/bin/bgzip -@ 8 > $O/fragments.tsv.gz && $ENV/bin/tabix -p bed $O/fragments.tsv.gz
/usr/bin/time -v -o $O/time_comp.txt $ENV/bin/python $W/src/singlet-py/claude__atac-spatial-benchmark-2ba7df__scripts/bench/atac_spatial/competitors_atac.py \
  $O/fragments.tsv.gz $G/genes/genes.gtf $O/comp > $O/comp.log 2>&1; echo "comparators exit=$?"; tail -5 $O/comp.log

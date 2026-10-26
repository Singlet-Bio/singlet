#!/bin/bash
#SBATCH --job-name=bp-ktools
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch kang_tools.sbatch <cohort-id> <n donors given to vireo> <harness dir>
# After kang_singlet.sbatch. Competitors on the SAME alignments (Singlet's tagged BAM, CB/UB) and the SAME cells (Singlet's
# called set): Scrublet and scDblFinder on the expression counts at their defaults; genotype doublets with cellsnp-lite mode 1a
# (vireo's documented pileup: common 1000G SNP list, --minMAF 0.1 --minCOUNT 20) followed by `vireo -N K`. vireo has no
# donor-count selection, so it is given the true K (an advantage over Singlet, which estimates it).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; BP=$W/bench-public; E=$W/envs/ambient; G=$W/envs/gpool
ID=$1; K=$2; H=$3; RUN=$BP/runs/$ID/singlet; D=$BP/work/$ID; mkdir -p $D/tools $D/vireo
PANEL=/mnt/projects/debruinz_project/cellarium/reference/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz
export HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl NUMBA_CACHE_DIR=$W/cache/numba-bp APPTAINER_BIND=/mnt/projects:/mnt/projects
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
T() { n=$1; shift; echo "== $n"; /usr/bin/time -v -o $D/tools/$n.time "$@" 2>&1 | tail -8 || true; grep -E "Elapsed \(wall|Maximum resident" $D/tools/$n.time; }
PATH=$E/bin:$PATH OMP_NUM_THREADS=1 python -I $H/kang_prep.py $RUN $D
PATH=$E/bin:$PATH OMP_NUM_THREADS=1 T scrublet python -I $H/scrublet_run.py $D
PATH=$E/bin:$PATH OMP_NUM_THREADS=1 T scdblfinder Rscript $H/barnyard_tools.R scdblfinder $D
PATH=$W/envs/pbench/bin:$PATH
echo "cellsnp-lite -s possorted.bam -b barcodes.txt -R PANEL -p 16 --minMAF 0.1 --minCOUNT 20 --gzip --cellTAG CB --UMItag UB" | tee $D/vireo/cmd.txt
T cellsnp apptainer exec $G/cellsnp-lite.sif cellsnp-lite -s $RUN/possorted.bam -b $D/barcodes.txt -O $D/vireo/cellsnp -R $PANEL \
  -p 16 --minMAF 0.1 --minCOUNT 20 --gzip --cellTAG CB --UMItag UB
echo "vireo -c cellsnp -o vireo -N $K -p 16 (defaults otherwise; randSeed 42)" | tee -a $D/vireo/cmd.txt
T vireo $G/venv/bin/vireo -c $D/vireo/cellsnp -o $D/vireo/out -N $K -p 16 --randSeed 42
lscpu | sed -n 's/^Model name: *//p' > $D/tools/node.txt; hostname >> $D/tools/node.txt
ls -la $D/tools $D/vireo/out

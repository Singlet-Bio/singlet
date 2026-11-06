#!/bin/bash
#SBATCH --job-name=rc-map
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --array=1-8%6
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Read-collapse study, stage 1 (one array task per sample): CPU engine run with a tagged BAM, then extraction and
# within-sample analyses (rc_extract.py). Scripts come from the bench/read-collapse branch.
# Test mode: --export=ALL,TEST=1,RES=<dir> maps pbmc2m once and extracts two halves as the samples tA and tB.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; RC=$W/jobs/read-collapse
REF=/mnt/projects/debruinz_project/cellarium/reference; PB=$W/pipeline-benchmark; DATA2=$W/jobs/fable/release/data
RES=${RES:-$RC/results/overnight}; T=${T:-16}
BIN=$W/jobs/fable/release/bin/singlet-gpu
BARE=$W/repos/singlet-py.git; SBR=bench/read-collapse
SCR=/tmp/$USER-rc-$SLURM_JOB_ID; mkdir -p $SCR/repo $RES; SREF=refs/rc/map-$SLURM_JOB_ID
trap 'git -C $BARE update-ref -d $SREF 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM
URL="$(git -C $BARE config remote.origin.url)"
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$SBR:$SREF ) 9>$BARE/.rc-fetch.lock || { echo "fetch failed"; exit 3; }
git -C $BARE archive $SREF bench/read_collapse | tar -x -C $SCR/repo
RCS=$SCR/repo/bench/read_collapse
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) scripts=$(git -C $BARE rev-parse --short $SREF) bin=$BIN commit=$(cat $(dirname $BIN)/commit-gpu) $(date -Is)"
export PATH=$W/envs/pbench/bin:$PATH SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_REF_BASE=$REF TMPDIR=$SCR
if [ "${TEST:-0}" = 1 ]; then S=pbmc2m; GENOME=GRCh38-2024-A
else read S CHEM TISSUE GENOME <<<"$(grep -v '^#' $RCS/samples.tsv | sed -n "${SLURM_ARRAY_TASK_ID}p")"; fi
[ -n "$S" ] || { echo "no sample for task $SLURM_ARRAY_TASK_ID"; exit 2; }
FQ=$PB/data/$S/$S.1fq; [ -f $FQ ] || FQ=$DATA2/$S/$S.1fq
G=$REF/$GENOME
O=$SCR/out; mkdir -p $O
echo "sample $S fq $FQ genome $GENOME"
time $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $O/ --threads $T \
  --pipeline --output-format 1pz --engine cpu --tagged-bam $SCR/tagged.bam > $SCR/run.log 2>&1
echo "singlet exit $?"; tail -3 $SCR/run.log | cut -c1-200
[ -s $SCR/tagged.bam ] || { echo "no BAM"; exit 4; }
NIN=$(awk -F'|' '/Number of input reads/{gsub(/[ \t]/,"",$2); print $2}' $O/star_Log.final.out)
awk 'NR%4==2' $O/star_Unmapped.out.mate1 > $SCR/unm.txt 2>/dev/null
echo "input reads $NIN, unmapped lines $(wc -l < $SCR/unm.txt)"
extract() {  # name rows
  local name=$1 rows=${2:-}
  mkdir -p $RES/$name; cp $O/star_Log.final.out $RES/$name/
  samtools view -@4 -F 0x900 $SCR/tagged.bam | awk -F'\t' 'BEGIN{OFS="\t"}{nh="";as="";cb="";ur="";for(i=12;i<=NF;i++){t=substr($i,1,2);if(t=="NH")nh=substr($i,6);else if(t=="AS")as=substr($i,6);else if(t=="CB")cb=substr($i,6);else if(t=="UR")ur=substr($i,6)}print $2,$3,$4,$6,nh,as,cb,ur,$10}' \
   | python3 $RCS/rc_extract.py --sample $name --gtf $G/genes/genes.gtf --out $RES/$name --unmapped $SCR/unm.txt --input-reads $NIN ${rows:+--rows $rows}
  echo "extract $name exit $?"
}
if [ "${TEST:-0}" = 1 ]; then
  NB=$(samtools view -c -F 0x900 $SCR/tagged.bam); H=$((NB/2))
  extract tA 0:$H; extract tB $H:$NB
else extract $S; fi
echo done $(date -Is)

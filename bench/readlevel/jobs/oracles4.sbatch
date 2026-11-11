#!/bin/bash
#SBATCH --job-name=rl-or4
#SBATCH --account=debruinz
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=450G
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# One packed node for two parts (replaces oracles3 / 804512):
#   B7 : finish scTE, SoloTE, te_eval for A375 and mESC (mESC comparators rebuilt once with the corrected mates)
#   B6 : the registered held-out run on GSE329598 (ORACLES_B6_B8.md, 97f3cc2b): fetch, hashtag labels, the frozen engine
#        configuration once, STAR-Fusion and Arriba on the same reads, then fusion_rank with and without the truth pair in the prior
set -uo pipefail
BR=${BR:-bench/readlevel-oracles}; NAME=${NAME:-rlo-13ee63ac}
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/readlevel; L=$W/logs/rl-or4-$SLURM_JOB_ID; mkdir -p $L $J/results
BIN=$J/bin/$NAME; TD=$W/data/readlevel/te; BARE=$W/repos/singlet-py.git; PY=$W/envs/pbench/bin/python
export PZ_SO=$W/venvs/forge/lib/python3.12/site-packages/singlet/_pz.cpython-312-x86_64-linux-gnu.so
export LD_LIBRARY_PATH=$W/envs/pbench/lib:${LD_LIBRARY_PATH:-}
SRC=$J/or4-src-$SLURM_JOB_ID; mkdir -p $SRC
( flock 9; git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/rl/or4-$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/rl/or4-$SLURM_JOB_ID bench/readlevel | tar -x -C $SRC; B=$SRC/bench/readlevel
echo "node=$(hostname) bench=$(git -C $BARE rev-parse refs/rl/or4-$SLURM_JOB_ID) $(date -Is)"

# ---------------- B7
O=$J/te/mesc_2c
if [ ! -f $TD/mesc_2c/.mates_fixed2 ]; then
  rm -rf $O/starsolo $O/scte* $O/solote* $O/comparators.log
  ln -sfn $TD/mesc_2c/ERR3981386_1.fastq.gz $TD/mesc_2c/10x/mesc_2c_S1_L001_R1_001.fastq.gz
  ln -sfn $TD/mesc_2c/ERR3981386_2.fastq.gz $TD/mesc_2c/10x/mesc_2c_S1_L001_R2_001.fastq.gz
  touch $TD/mesc_2c/.mates_fixed2; echo "mesc_2c: comparator outputs from swapped mates cleared"
fi
te() {   # NAME threads
  local n=$1 O=$J/te/$1 E=$J/runs/te/$1/out
  [ -d $E ] || { echo "no engine run for $n"; return 1; }
  [ -s $O/starsolo/Aligned.sortedByCoord.out.bam.bai ] || rm -rf $O/starsolo
  SLURM_JOB_ID=${SLURM_JOB_ID}t$n SLURM_CPUS_PER_TASK=$2 NAME=$n BIN=$BIN SKIP_ENGINE=1 bash $B/te/jobs/te_bench.sbatch > $L/te_bench-$n.log 2>&1
  if [ $n = a375_10x ]; then
    $PY -I $B/te/te_eval.py bulk --engine $E --scte $O/scte.h5ad --solote $O/solote --truth $O/longread_truth.tsv --out $J/results/b7_a375.json
  else
    $PY -I $B/te/te_eval.py twoc --engine $E --scte $O/scte.h5ad --solote $O/solote --solo-gene $O/starsolo/Solo.out/Gene/filtered --out $J/results/b7_mesc.json
  fi
}
( te a375_10x 12 ) > $L/te-a375.log 2>&1 & PA=$!
( te mesc_2c 14 ) > $L/te-mesc.log 2>&1 & PM=$!

# ---------------- B6 held-out
(
  set -uo pipefail
  S=ku812_hmc; D=$W/data/readlevel/holdout/$S; F=$J/fusion/$S; mkdir -p $D/10x $D/hto $F
  export PATH=$W/envs/sratools/bin:$PATH
  fq() { local r=$1 o=$2; [ -f $o/$r.DONE ] && return 0
    rm -f $o/${r}_*.fastq $o/${r}_*.fastq.gz   # a fetch cut off by the end of an earlier job (hofetch in 804425) starts again
    fasterq-dump -f --split-files --include-technical -e 6 -t /tmp/$USER-fq-$SLURM_JOB_ID -O $o $r || return 1
    for f in $o/${r}_*.fastq; do gzip -1 $f; done; touch $o/$r.DONE; }
  mates() { local d=$1 r=$2 best=0 l; M1=; M2=   # the barcode read is 28 bases; the cDNA or feature read is the longest other file
    for f in $d/${r}_*.fastq.gz; do l=$(zcat $f | head -2 | tail -1 | awk '{print length($0)}'); echo "$(basename $f) $l"
      if [ "$l" = 28 ]; then M1=$f; elif [ "$l" -gt "$best" ]; then best=$l; M2=$f; fi; done
    [ -n "$M1" ] && [ -n "$M2" ] || { echo "mates of $r not found"; return 1; }; }
  lane=0; for r in SRR38330751 SRR38330752; do lane=$((lane+1)); fq $r $D || { echo "fetch $r failed"; exit 1; }; mates $D $r || exit 1
    ln -sfn $M1 $D/10x/${S}_S1_L00${lane}_R1_001.fastq.gz; ln -sfn $M2 $D/10x/${S}_S1_L00${lane}_R2_001.fastq.gz; done
  H1=; H2=; for r in SRR38330749 SRR38330750; do fq $r $D/hto || { echo "fetch $r failed"; exit 1; }; mates $D/hto $r || exit 1; H1=${H1:+$H1,}$M1; H2=${H2:+$H2,}$M2; done
  [ -s $D/pacbio.csv.gz ] || curl -sSfL -o $D/pacbio.csv.gz https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9707nnn/GSM9707175/suppl/GSM9707175_Leukemic_cell_lines_long_read_count_table.csv.gz
  TRL=/mnt/projects/debruinz_project/kodumags/cellranger-9.0.1/lib/python/cellranger/barcodes/translation/3M-february-2018_NXT.txt.gz
  $PY -I $B/fusion/hto_labels.py --r1 $H1 --r2 $H2 \
    --whitelist $W/pipeline-benchmark/whitelists/3M-february-2018.txt --translate $TRL --pacbio $D/pacbio.csv.gz --out $F/labels.tsv --summary $J/results/b6_holdout_labels.json
  rc=$?; [ $rc = 0 ] || { echo "labels exit $rc: stop before any caller"; exit 1; }
  # the frozen engine configuration, once
  PBIN=$J/bin/chimproto-68d44117; BL=$W/envs/fusion/var/lib/arriba/blacklist_hg38_GRCh38_v2.5.1.tsv.gz; RC=$J/fusion/chimeric_recurrent_GRCh38.tsv
  ( SLURM_JOB_ID=${SLURM_JOB_ID}he SLURM_CPUS_PER_TASK=8 BIN=$PBIN TAG=holdout-frozen SAMPLE=$S FQDIR=$D/10x \
      EXTRA="--chimeric --chimeric-blacklist $BL --chimeric-recurrent $RC" SINGLET_CHIM_SOFTCLIP_MIN=15 SINGLET_KEEP_CHIMERIC_JUNCTION=1 \
      bash $J/run_engine.sbatch > $L/holdout-engine.log 2>&1; echo "engine exit $?" ) &
  ( SLURM_JOB_ID=${SLURM_JOB_ID}hc SLURM_CPUS_PER_TASK=12 FQDIR=$D/10x SAMPLE=$S bash $B/fusion/jobs/comparators.sbatch > $L/holdout-comparators.log 2>&1; echo "comparators exit $?" ) &
  wait
  $PY -I $B/fusion/fusion_eval.py names --starfusion $F/starfusion/star-fusion.fusion_predictions.tsv --arriba $F/arriba/fusions.tsv --out $F/names.txt
  $PY -I $B/fusion/fusion_eval.py readmap --names $F/names.txt --r1 $(ls $D/10x/*_R1_001.fastq.gz | sort | paste -sd,) --cb-len 16 --umi-len 12 --out $F/readmap.tsv
  GTF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/genes/genes.gtf; MT=$J/fusion/mitelman/MCGENE.TXT.DATA
  for lto in "" "--leave-truth-out"; do
    $PY -I $B/fusion/fusion_rank.py $lto --engine $J/runs/holdout-frozen/$S/out/chimeric_junctions.tsv.gz --starfusion $F/starfusion/star-fusion.fusion_predictions.tsv \
      --arriba $F/arriba/fusions.tsv --readmap $F/readmap.tsv --labels $F/labels.tsv --truth "Ku812:BCR--ABL1" --gtf $GTF --mitelman $MT \
      --out $J/results/b6_holdout_rank${lto:+_lto}.json > $L/holdout-rank${lto:+_lto}.log 2>&1; echo "rank${lto:+_lto} exit $?"
  done
  grep -a "reads into the second pass" $J/runs/holdout-frozen/$S/run.log
) > $L/holdout.log 2>&1 & PH=$!

for x in te-a375:$PA te-mesc:$PM holdout:$PH; do wait ${x#*:}; echo "${x%%:*} exit $? $(date -Is)"; done
ls -l $J/results | grep -E "b7|holdout"

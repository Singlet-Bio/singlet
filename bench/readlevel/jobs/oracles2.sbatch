#!/bin/bash
#SBATCH --job-name=rl-or2
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=500G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Launch oracles B6/B7/B8, part 2 (after oracles1.sbatch), packed on one node. Parts run side by side, each logs to $L/<part>.log:
#   b6   : cell-line labels, comparator read map, fusion_eval on both Penter mixes -> $J/results/b6_<mix>.json
#   b8   : matched-molecule reads per UMI and the full-set check on the new engine run -> $J/results/b8_*.json
#   te-* : te_bench.sbatch per dataset (scTE + SoloTE on the STARsolo BAM, A375 nanopore truth), then te_eval -> $J/results/b7_*.json
# Any engine run oracles1 left unfinished is rerun here first (run_engine refuses to overwrite a finished one).
set -uo pipefail
BR=${BR:-bench/readlevel-oracles}; NAME=${NAME:-rlo-13ee63ac}
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/readlevel; L=$W/logs/rl-or2-$SLURM_JOB_ID; mkdir -p $L $J/results
BIN=$J/bin/$NAME; P=$W/data/readlevel/penter2024; TD=$W/data/readlevel/te; BARE=$W/repos/singlet-py.git
PY=$W/envs/pbench/bin/python; export PZ_SO=$W/venvs/forge/lib/python3.12/site-packages/singlet/_pz.cpython-312-x86_64-linux-gnu.so
export LD_LIBRARY_PATH=$W/envs/pbench/lib:${LD_LIBRARY_PATH:-}
SCR=/tmp/$USER-or2-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
( flock 9; git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/rl/or2-$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/rl/or2-$SLURM_JOB_ID bench/readlevel | tar -x -C $SCR; B=$SCR/bench/readlevel
echo "node=$(hostname) bench=$(git -C $BARE rev-parse refs/rl/or2-$SLURM_JOB_ID) bin=$(cat $BIN/commit-gpu) $(date -Is)"
$PY -I -c "import importlib.machinery as m, importlib.util as u, os; s=os.environ['PZ_SO']; l=m.ExtensionFileLoader('singlet._pz', s); \
  x=u.module_from_spec(u.spec_from_file_location('singlet._pz', s, loader=l)); l.exec_module(x); print('pz ok')" || echo "PZ_SO does not load"

eng() {   # TAG SAMPLE threads [GENOME FQDIR EXTRA]: engine run unless finished
  local o=$J/runs/$1/$2; [ -d $o/out ] && return 0; rm -rf $o
  SLURM_JOB_ID=${SLURM_JOB_ID}$2 SLURM_CPUS_PER_TASK=$3 BIN=$BIN TAG=$1 SAMPLE=$2 GENOME=${4:-GRCh38-2024-A} FQDIR=${5:-} EXTRA=${6:-} \
    SINGLET_KEEP_CHIMERIC_JUNCTION=1 bash $J/run_engine.sbatch > $L/engine-$2.log 2>&1
}

( # B6
  for m in mix1to1p5 mix1to15; do
    ( eng chim $m 8 GRCh38-2024-A $P/$m --chimeric
      E=$J/runs/chim/$m/out; F=$J/fusion/$m; R=$J/results
      $PY -I $B/fusion/fusion_labels.py --engine $E --out $F/labels.tsv --summary $R/b6_${m}_labels.json || exit 1
      $PY -I $B/fusion/fusion_eval.py names --starfusion $F/starfusion/star-fusion.fusion_predictions.tsv --arriba $F/arriba/fusions.tsv --out $F/names.txt
      $PY -I $B/fusion/fusion_eval.py readmap --names $F/names.txt --r1 $(ls $P/$m/*_R1_001.fastq.gz | sort | paste -sd,) --cb-len 16 --umi-len 10 --out $F/readmap.tsv
      $PY -I $B/fusion/fusion_eval.py eval --engine $E/chimeric_junctions.tsv.gz --starfusion $F/starfusion/star-fusion.fusion_predictions.tsv \
        --arriba $F/arriba/fusions.tsv --readmap $F/readmap.tsv --labels $F/labels.tsv --truth "K562:BCR--ABL1,NUP214--XKR3;Kasumi-1:RUNX1--RUNX1T1" \
        --out $R/b6_$m.json > $L/b6-eval-$m.log 2>&1; echo "b6 $m eval exit $?"
    ) > $L/b6-$m.log 2>&1 &
  done; wait
) & P6=$!

( # B8
  M=$J/runs/rpu-mol/pbmc1k_v3; PB=$W/pipeline-benchmark/data/pbmc1k_v3; MI=$W/data/readlevel/pbmc1k_v3/pbmc_1k_v3_molecule_info.h5
  if [ ! -d $M/out ] || [ ! -s $M/molecules.tsv.gz ]; then
    rm -rf $M; mkdir -p $M
    SLURM_JOB_ID=${SLURM_JOB_ID}er SLURM_CPUS_PER_TASK=6 BIN=$BIN TAG=rpu-mol SAMPLE=pbmc1k_v3 FQ=$PB/pbmc1k_v3.1fq \
      SINGLET_MOLECULE_DUMP=$M/molecules.tsv.gz bash $J/run_engine.sbatch > $L/engine-pbmc1k.log 2>&1
  fi
  $PY -I $B/rpu_vs_molinfo.py --hist $M/out/reads_per_umi_hist.tsv.gz --molinfo $MI --cr-filtered $PB/cellranger_pbmc_1k_v3_filtered_feature_bc_matrix.h5 \
    --calls $M/out/cell_calls.tsv --out $J/results/b8_full.json
  $PY -I $B/rpu_matched.py --dump $M/molecules.tsv.gz --hist $M/out/reads_per_umi_hist.tsv.gz --molinfo $MI \
    --cr-filtered $PB/cellranger_pbmc_1k_v3_filtered_feature_bc_matrix.h5 --calls $M/out/cell_calls.tsv --out $J/results/b8_matched.json
) > $L/b8.log 2>&1 & P8=$!

# mESC: 804363 used the 101 bp barcode read as cDNA (the old length rule); clear everything built from that once
M=$TD/mesc_2c
if [ ! -f $M/.mates_fixed ]; then
  rm -rf $J/te/mesc_2c/starsolo $J/runs/te/mesc_2c $W/data/readlevel/1fq/mesc_2c.1fq $M/10x
  mkdir -p $M/10x; ln -sfn $M/ERR3981386_1.fastq.gz $M/10x/mesc_2c_S1_L001_R1_001.fastq.gz; ln -sfn $M/ERR3981386_2.fastq.gz $M/10x/mesc_2c_S1_L001_R2_001.fastq.gz
  touch $M/.mates_fixed; echo "mesc_2c: outputs from swapped mates cleared"
fi

te() {   # NAME GENOME threads
  local n=$1 F=$TD/$1/10x O=$J/te/$1
  eng te $n $3 $2 $F
  [ -s $O/starsolo/Aligned.sortedByCoord.out.bam.bai ] || rm -rf $O/starsolo   # a BAM without its index is from a cut-off run
  SLURM_JOB_ID=${SLURM_JOB_ID}t$n SLURM_CPUS_PER_TASK=$3 NAME=$n BIN=$BIN bash $J/te_bench.sbatch > $L/te_bench-$n.log 2>&1
  local E=$J/runs/te/$n/out
  if [ $n = a375_10x ]; then
    $PY -I $B/te/te_eval.py bulk --engine $E --scte $O/scte.h5ad --solote $O/solote --truth $O/longread_truth.tsv --out $J/results/b7_a375.json
  else
    $PY -I $B/te/te_eval.py twoc --engine $E --scte $O/scte.h5ad --solote $O/solote --solo-gene $O/starsolo/Solo.out/Gene/filtered --out $J/results/b7_mesc.json
  fi
}
( te a375_10x GRCh38-2024-A 14 ) > $L/te-a375.log 2>&1 & PA=$!
( te mesc_2c GRCm39-2024-A 12 ) > $L/te-mesc.log 2>&1 & PM=$!

for x in b6:$P6 b8:$P8 te-a375:$PA te-mesc:$PM; do wait ${x#*:}; echo "${x%%:*} exit $? $(date -Is)"; done
ls -l $J/results

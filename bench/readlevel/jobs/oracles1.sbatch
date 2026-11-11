#!/bin/bash
#SBATCH --job-name=rl-or1
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=500G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Launch oracles B6/B7/B8, part 1, packed on one node. Parts run side by side and each logs to $L/<part>.log:
#   build  : the engine at BR (frontier + feat/chimeric-junctions + feat/te-counts), then
#            B6 engine --chimeric on both Penter mixes, B8 engine on pbmc1k_v3 with the per-molecule dump, then B7 engine on A375 and mESC
#   comp   : B6 STAR-Fusion + Arriba on both Penter mixes
#   fetch  : B7 data still missing after 804124 (mESC R2 resume, A375 nanopore reads)
#   solo   : B7 STARsolo BAMs for scTE/SoloTE (A375 now, mESC once its reads are complete)
set -uo pipefail
BR=${BR:-bench/readlevel-oracles}; NAME=${NAME:-rlo-13ee63ac}
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/readlevel; L=$W/logs/rl-or1-$SLURM_JOB_ID; mkdir -p $L
REF=/mnt/projects/debruinz_project/cellarium/reference; WL=$W/pipeline-benchmark/whitelists; ENV=$W/envs/tebench
BIN=$J/bin/$NAME; P=$W/data/readlevel/penter2024; TD=$W/data/readlevel/te
echo "node=$(hostname) cpus=$SLURM_CPUS_ON_NODE br=$BR name=$NAME $(date -Is)"

solo() {   # NAME_TE GENOME R1 R2 WL UMIL threads: STARsolo as te_bench.sbatch runs it (te_bench skips it when the BAM exists)
  local n=$1 g=$REF/$2 r1=$3 r2=$4 wl=$5 ul=$6 t=$7 o=$J/te/$1
  [ -s $o/starsolo/Aligned.sortedByCoord.out.bam.bai ] && { echo "$n BAM present"; return 0; }
  mkdir -p $o/starsolo
  $ENV/bin/STAR --runThreadN $t --genomeDir $g/star_2.7.11b --readFilesIn $r2 $r1 --readFilesCommand zcat --soloType CB_UMI_Simple \
    --soloCBwhitelist $wl --soloUMIlen $ul --soloBarcodeReadLength 0 --soloFeatures Gene --soloCellFilter EmptyDrops_CR \
    --soloUMIdedup 1MM_CR --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR --clipAdapterType CellRanger4 \
    --outFilterMultimapNmax 100 --winAnchorMultimapNmax 100 --outSAMmultNmax 1 --outSAMtype BAM SortedByCoordinate --limitBAMsortRAM 60000000000 \
    --outSAMattributes NH HI AS nM CR CY UR UY CB UB GX GN sS sQ sM --outFileNamePrefix $o/starsolo/ > $o/starsolo.log 2>&1 || { echo "$n STAR failed"; return 1; }
  $ENV/bin/samtools index -@ $t $o/starsolo/Aligned.sortedByCoord.out.bam && echo "$n BAM done $(date -Is)"
}

( # fetch
  SLURM_JOB_ID=${SLURM_JOB_ID}f WHAT="mesc_2c a375_ont" bash $J/fetch_te.sbatch
) > $L/fetch.log 2>&1 & PF=$!

( # comparators, both mixes side by side
  for m in mix1to1p5 mix1to15; do
    SLURM_JOB_ID=${SLURM_JOB_ID}c$m SLURM_CPUS_PER_TASK=8 SAMPLE=$m FQDIR=$P/$m bash $J/comparators.sbatch > $L/comp-$m.log 2>&1 &
  done; wait
) & PC=$!

( # STARsolo: A375 now; mESC after its reads are complete
  solo a375_10x GRCh38-2024-A $TD/a375_10x/SRR21492169_2.fastq.gz $TD/a375_10x/SRR21492169_3.fastq.gz $WL/3M-february-2018.txt 12 8
  while kill -0 $PF 2>/dev/null; do sleep 60; done
  [ -f $TD/mesc_2c/DONE ] || { echo "mesc_2c not fetched"; exit 1; }
  r1=$TD/mesc_2c/ERR3981386_1.fastq.gz; r2=$TD/mesc_2c/ERR3981386_2.fastq.gz
  l1=$(zcat $r1 | head -2 | tail -1 | awk '{print length($0)}'); [ "$l1" -gt 50 ] && { t=$r1; r1=$r2; r2=$t; }
  solo mesc_2c GRCm39-2024-A $r1 $r2 $WL/3M-february-2018.txt 12 8
) > $L/solo.log 2>&1 & PS=$!

( # build, then the engine runs
  if [ ! -x $BIN/singlet-gpu ]; then
    SLURM_JOB_ID=${SLURM_JOB_ID}b SLURM_CPUS_PER_TASK=16 TESTS=0 BR=$BR NAME=$NAME bash $J/build_arch.sbatch > $L/build.log 2>&1 || { echo "build failed"; tail -30 $L/build.log; exit 1; }
  fi
  echo "build ok $(cat $BIN/commit-gpu) $(date -Is)"
  for m in mix1to1p5 mix1to15; do
    SLURM_JOB_ID=${SLURM_JOB_ID}e$m SLURM_CPUS_PER_TASK=8 BIN=$BIN TAG=chim SAMPLE=$m FQDIR=$P/$m EXTRA=--chimeric \
      SINGLET_KEEP_CHIMERIC_JUNCTION=1 bash $J/run_engine.sbatch > $L/engine-$m.log 2>&1 &
  done
  mkdir -p $J/runs/rpu-mol/pbmc1k_v3
  SLURM_JOB_ID=${SLURM_JOB_ID}er SLURM_CPUS_PER_TASK=6 BIN=$BIN TAG=rpu-mol SAMPLE=pbmc1k_v3 FQ=$W/pipeline-benchmark/data/pbmc1k_v3/pbmc1k_v3.1fq \
    SINGLET_MOLECULE_DUMP=$J/runs/rpu-mol/pbmc1k_v3/molecules.tsv.gz bash $J/run_engine.sbatch > $L/engine-pbmc1k.log 2>&1 &
  wait
  echo "B6/B8 engine runs done $(date -Is)"
  # B7 engine: A375 now; mESC once fetched (10x names as te_bench.sbatch makes them)
  F=$TD/a375_10x/10x; mkdir -p $F
  ln -sfn $TD/a375_10x/SRR21492169_2.fastq.gz $F/a375_10x_S1_L001_R1_001.fastq.gz; ln -sfn $TD/a375_10x/SRR21492169_3.fastq.gz $F/a375_10x_S1_L001_R2_001.fastq.gz
  SLURM_JOB_ID=${SLURM_JOB_ID}ta SLURM_CPUS_PER_TASK=16 BIN=$BIN TAG=te SAMPLE=a375_10x FQDIR=$F GENOME=GRCh38-2024-A bash $J/run_engine.sbatch > $L/engine-a375.log 2>&1 &
  while kill -0 $PF 2>/dev/null; do sleep 60; done
  if [ -f $TD/mesc_2c/DONE ]; then
    r1=$TD/mesc_2c/ERR3981386_1.fastq.gz; r2=$TD/mesc_2c/ERR3981386_2.fastq.gz
    l1=$(zcat $r1 | head -2 | tail -1 | awk '{print length($0)}'); [ "$l1" -gt 50 ] && { t=$r1; r1=$r2; r2=$t; }
    F=$TD/mesc_2c/10x; mkdir -p $F; ln -sfn $r1 $F/mesc_2c_S1_L001_R1_001.fastq.gz; ln -sfn $r2 $F/mesc_2c_S1_L001_R2_001.fastq.gz
    SLURM_JOB_ID=${SLURM_JOB_ID}tm SLURM_CPUS_PER_TASK=12 BIN=$BIN TAG=te SAMPLE=mesc_2c FQDIR=$F GENOME=GRCm39-2024-A bash $J/run_engine.sbatch > $L/engine-mesc.log 2>&1 &
  fi
  wait
) > $L/engine.log 2>&1 & PE=$!

for x in fetch:$PF comp:$PC solo:$PS engine:$PE; do wait ${x#*:}; echo "${x%%:*} exit $? $(date -Is)"; done

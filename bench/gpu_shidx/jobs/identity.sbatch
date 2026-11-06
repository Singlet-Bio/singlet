#!/bin/bash
#SBATCH --job-name=gshidx-ident
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=300G
#SBATCH --time=02:00:00
#SBATCH --exclude=g050
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Identity of the shared resident genome index (gpu/bold-shidx) on pbmc1k_v3, T=7, level 5 + ingest (the throughput harness' settings).
# Legs: the CPU engine of the new binary, b0 alone on the GPU, then two processes at once at 2 slots sharing one holder (started lazily by the first
# process), then the same two processes under MPS (also the CUDA IPC check under MPS). Every GPU output is compared with the CPU output and with b0
# (the gate's pair comparison, copied from the throughput harness). Verdict: results/overnight/identity-<job>/verdict.txt
# Parameters: SH (directory of the new binary), B0 (directory of b0). Node: g051 or g052 (g050 is the timing node).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
SH=${SH:-$W/jobs/gpu-shidx/bin/shidx}; B0=${B0:-$W/jobs/gpu-bold/bin/b0}; T=7; s=pbmc1k_v3
RES0=$W/jobs/gpu-shidx/results/overnight/identity-$SLURM_JOB_ID; mkdir -p $RES0; V=$RES0/verdict.txt
SCR=/tmp/$USER-shid-$SLURM_JOB_ID; mkdir -p $SCR/repo $SCR/data
BARE=$W/repos/singlet-py.git; SREF=refs/gshidx/id-$SLURM_JOB_ID
HOLDER_GENOME=$REF/GRCh38-2024-A/star_2.7.11b
cleanup() { kill $(jobs -p) 2>/dev/null; $SH/singlet-gpu gpu-holder --stop --genome-dir $HOLDER_GENOME --device 0 2>/dev/null; echo quit | nvidia-cuda-mps-control 2>/dev/null; git -C $BARE update-ref -d $SREF 2>/dev/null; rm -rf "${SCR:?}"; }
trap cleanup EXIT INT TERM
URL="$(git -C $BARE config remote.origin.url)"
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/bench/gpu-batch1:$SREF ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive $SREF bench/star_accel/out_equiv.py | tar -x -C $SCR/repo
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) new=$(cat $SH/commit-gpu) sha256=$(sha256sum $SH/singlet-gpu | cut -c1-64) b0=$(cat $B0/commit-gpu) $(date -Is)" | tee $V
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a $V
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
[ -e $SH/../whitelists ] || ln -s $PB/whitelists $SH/../whitelists 2>/dev/null
src=$PB/data/$s/$s.1fq; [ -f $src ] || src=$W/jobs/fable/release/data/$s/$s.1fq
cp $src $SCR/data/$s.1fq
G=$REF/GRCh38-2024-A

EXPECT_DIFF=" ${EXPECT_DIFF:-pileup_stats.json read_stats.tsv star_Log.std.out summary.json} "
KNOWN_nuclei2k_v2=${KNOWN_nuclei2k_v2-nonhost_reads.fq}   # the known non-host writer bug of the gate (E15 section 5.7)
READF="nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2"
prune() { find $1 \( -name '*.bam' -o -name '*.sam' -o -name '*.cram' -o -name '*.fq*' -o -name '*.fastq*' -o -name '*.1fq' -o -name '.gpu_handoff*' -o -name '*Unmapped*' \
                     -o -name '_STARtmp*' -o -name '*.reads' -o -name '*.reads.q' \) -prune -exec rm -rf {} + 2>/dev/null; }
readhash() { for f in $READF; do
    if [ -f $1/$f ]; then echo "$f $(paste - - - - < $1/$f | LC_ALL=C sort -S 4G --parallel=4 -T $SCR | sha256sum | cut -c1-16) $(( $(wc -l < $1/$f) / 4 ))"
    else echo "$f absent 0"; fi; done; }

# compare_out LABEL A_OUT A_CRS A_RH B_OUT B_CRS B_RH DETAILFILE   -> 0 when identical, 1 when not (same checks as the gate's pair comparison)
compare_out() {
  local lab=$1 a=$2 ca=$3 rha=$4 b=$5 cb=$6 rhb=$7 d=$8 ok=1 f
  echo "=== $lab: $b vs $a" >> $d
  if [ ! -d "$a" ]; then echo "  no reference ($a)" >> $d; return 2; fi
  python3 $SCR/repo/bench/star_accel/out_equiv.py $a $b > $d.equiv 2>&1
  grep -a '^  DIFF\|^  only in\|^files:\|^RESULT' $d.equiv | cut -c1-220 >> $d
  local UNX=""
  for f in $(grep -a '^  DIFF ' $d.equiv | sed -E 's/^  DIFF ([^ :]+).*/\1/'); do case "$EXPECT_DIFF" in *" $f "*) ;; *) UNX="$UNX $f" ;; esac; done
  grep -aq '^  DIFF provenance.json: /references/whitelist/name: [^;]*$' $d.equiv && UNX=${UNX/ provenance.json/}
  for f in $(grep -a '^  only in ' $d.equiv | sed -E 's/^  only in [ab]: //'); do UNX="$UNX only:$f"; done
  grep -aq '^files:' $d.equiv || UNX="$UNX out_equiv-failed"
  [ -n "$UNX" ] && { echo "  out_equiv UNEXPECTED:$UNX" >> $d; ok=0; }
  for f in Barcodes.stats Gene/Features.stats Gene/Summary.csv Gene/UMIperCellSorted.txt Gene/raw/barcodes.tsv Gene/raw/features.tsv Gene/raw/matrix.mtx \
           Gene/filtered/barcodes.tsv Gene/filtered/features.tsv Gene/filtered/matrix.mtx; do
    cmp -s $a/star_Solo.out/$f $b/star_Solo.out/$f || { echo "  star_Solo.out/$f DIFFERS" >> $d; ok=0; }
  done
  cmp -s <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out) || { echo "  star_Log.final.out STATISTICS DIFFER" >> $d; ok=0; }
  cmp -s $a/star_SJ.out.tab $b/star_SJ.out.tab || { echo "  star_SJ.out.tab DIFFERS" >> $d; ok=0; }
  if [ -f "$rha" ] && [ -f "$rhb" ]; then
    local kv=KNOWN_$s; local KN=" ${!kv:-} "
    for f in $READF; do
      local ha hb; ha=$(awk -v f=$f '$1==f{print $2" "$3}' $rha); hb=$(awk -v f=$f '$1==f{print $2" "$3}' $rhb)
      if [ "$ha" = "$hb" ]; then :
      elif [[ "$KN" == *" $f "* ]] && [ -n "$ha" ] && [ "${ha##* }" = "${hb##* }" ]; then echo "  $f: other records, same count (known non-host writer bug)" >> $d
      else echo "  $f: RECORDS DIFFER ($ha vs $hb)" >> $d; ok=0; fi
    done
  else echo "  read files: not compared (no hashes)" >> $d; ok=0; fi
  if [ -f "$ca" ] && [ -f "$cb" ]; then
    { cmp -s <(head -1 $ca) <(head -1 $cb) && cmp -s <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort); } || { echo "  CellReads.stats ROWS DIFFER" >> $d; ok=0; }
  else echo "  CellReads.stats: not compared" >> $d; ok=0; fi
  [ $ok = 1 ] && { echo "  IDENTICAL" >> $d; return 0; } || { echo "  NOT IDENTICAL" >> $d; return 1; }
}



# run LABEL BINDIR ENGINE(cpu|gpu) [slots] -> outputs in $SCR/LABEL/out, log in $RES0/LABEL
run() {
  local lab=$1 bd=$2 eng=$3 sl=${4:-1} D=$SCR/$1 O=$RES0/$1 ENGARG="" XE=()
  mkdir -p $D/out $O
  [ $eng = gpu ] && { ENGARG="--engine gpu"; XE=(SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_SLOTS=$sl); }
  ( while [ ! -e $D/out/star_Solo.out/Gene/CellReads.stats ]; do sleep 0.2; done; ln $D/out/star_Solo.out/Gene/CellReads.stats $D/CellReads.stats ) 2>/dev/null &
  local WATCH=$!
  ( cd $D && env "${XE[@]}" /usr/bin/time -v -o $O/time.txt timeout -k 30 3000 $bd/singlet-gpu $SCR/data/$s.1fq --ref-base $REF --genome-dir $G/star_2.7.11b \
      --exons $G/genes/genes.gtf --out-prefix $D/out/ --threads $T --pipeline --output-format 1pz $ENGARG > $O/run.log 2>&1 )
  local rc=$?; kill $WATCH 2>/dev/null
  cp $D/out/*Log.final.out $D/out/*Log.out $O/ 2>/dev/null
  echo "$lab: exit $rc" | tee -a $V
  return $rc
}
keep() {   # LABEL: read hashes, prune
  readhash $SCR/$1/out > $SCR/$1/readhash.txt; prune $SCR/$1/out
}
cmpv() {   # LABEL REFLABEL
  local d=$RES0/$1.vs.$2.txt; : > $d
  compare_out "$1 vs $2" $SCR/$2/out $SCR/$2/CellReads.stats $SCR/$2/readhash.txt $SCR/$1/out $SCR/$1/CellReads.stats $SCR/$1/readhash.txt $d
  local r=$?; echo "  $1 vs $2: $(grep -a 'IDENTICAL' $d | tail -1 | tr -d ' ')" | tee -a $V; return $r
}
shared_note() {   # LABEL: what the sharing did in this run's log
  local l=$RES0/$1/star_Log.out
  echo "  $1: shared-index lines: imported=$(grep -ac 'shared index: imported' $l 2>/dev/null) private=$(grep -ac 'shared index:.*private upload' $l 2>/dev/null) mode: $(grep -ao 'mode [a-z]*' $l 2>/dev/null | sort | uniq -c | tr '\n' ' ')" | tee -a $V
  grep -a 'shared index:' $l 2>/dev/null | cut -c1-300 | sed 's/^/    /' | tee -a $V
}

# 1. CPU engine of the new binary, 2. b0 on the GPU
run cpu $SH cpu; keep cpu
run b0 $B0 gpu 1; keep b0
cmpv b0 cpu
# 3. two processes at once, one holder (started by the first process), 2 slots
export SINGLET_GPU_SHARED_INDEX=1 SINGLET_GPU_SHARED_INDEX_IDLE_S=120
for lab in sh1 sh2; do run $lab $SH gpu 2 & done; wait
for lab in sh1 sh2; do keep $lab; cmpv $lab cpu; cmpv $lab b0; shared_note $lab; done
cat /tmp/singlet_gpu_holder.*.log 2>/dev/null | cut -c1-300 | sed 's/^/  holder log: /' | tee -a $V
$SH/singlet-gpu gpu-holder --stop --genome-dir $HOLDER_GENOME --device 0; sleep 10
# 4. the same under MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/$USER-mps-$SLURM_JOB_ID/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/$USER-mps-$SLURM_JOB_ID/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "MPS daemon started" | tee -a $V
sleep 3
for lab in mps1 mps2; do run $lab $SH gpu 2 & done; wait
for lab in mps1 mps2; do keep $lab; cmpv $lab cpu; cmpv $lab b0; shared_note $lab; done
cat /tmp/singlet_gpu_holder.*.log 2>/dev/null | tail -5 | cut -c1-300 | sed 's/^/  holder log: /' | tee -a $V
cp $CUDA_MPS_LOG_DIRECTORY/*.log $RES0/ 2>/dev/null
$SH/singlet-gpu gpu-holder --stop --genome-dir $HOLDER_GENOME --device 0
if [ "$(grep -ac ': IDENTICAL$\|:IDENTICAL$' $V)" != 9 ] || grep -aq 'NOTIDENTICAL\|: exit [1-9]' $V; then echo "IDENTITY: FAIL" | tee -a $V; else echo "IDENTITY: PASS (every comparison identical, every run exit 0)" | tee -a $V; fi
echo done $(date -Is) | tee -a $V

#!/bin/bash
#SBATCH --job-name=bp-dl-bench
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
#   sbatch dl_bench.sbatch REP "MODE MODE ..." [DIGEST]
#     MODE  singlet_remote | singlet_prefetch_local | singlet_orch_io | singlet_lossless | fasterq_dump | prefetch_fasterq_dump
#           | aws_s3 | ena_https
#     DIGEST  1 (default for REP 1) decode every download and score it against dl_reference.sbatch; 0 timing only
# Every mode fetches the same four runs (runs.sh) as four concurrent streams (one run each) from a Clipper compute node into
# node-local /tmp, from a fresh HOME so no sra-tools cache is shared between modes. Per stream: wall seconds from the start of
# the stream to the last byte decoded (the clock stops before any scoring). Modes run one after the other, in the order given
# (the caller alternates it between repetitions).
#   singlet_remote         singlet download SRR                     (VDB streaming, the 2026 campaign path)
#   singlet_prefetch_local prefetch SRR; singlet download X.sra     (the atlas re-run path)
#   singlet_orch_io        the I/O stage of feat/campaign-orchestrator (stages.io_stage): head download of 1 M reads, screen-runs,
#                          prefetch, local download, screen-runs on the full run
#   singlet_lossless       prefetch; singlet download --no-trim --quality full X.sra    (to score decoded reads against the source)
#   fasterq_dump           fasterq-dump --split-files -e 4 SRR      (sra-tools, fetches the run itself)
#   prefetch_fasterq_dump  prefetch SRR; fasterq-dump --split-files -e 4 X.sra
#   aws_s3                 curl the .sra from the sra-pub-run-odp bucket (bytes only; scored by sha256 against prefetch's file)
#   ena_https              curl the run's FASTQ files from ENA over HTTPS
set -u
source ${BP_SRC:-/mnt/projects/debruinz_project/debruinz/singlet/bench-public/src/singlet}/bench/public/harness/perf_common.sh
source $SRC/bench/public/harness/download/runs.sh
REP=$1; MODES=$2; DIGEST=${3:-$([ "$REP" = 1 ] && echo 1 || echo 0)}
verify_bin
export PATH=$ENVDIR/bin:$PATH
OUTB=$BP/download/rep$REP; mkdir -p $OUTB
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
REFD=$BP/download/reference
echo "[dl] rep $REP modes: $MODES digest=$DIGEST node=$(hostname) cpu=$(cpu_model) $(date -Is)"
echo "[dl] prefetch $(prefetch --version | tail -1), fasterq-dump $(fasterq-dump --version | tail -1), singlet $BIN_COMMIT"

stream_fn() {  # MODE SRR D -> prints nothing; leaves $D/$SRR.t (start end rc) and $D/$SRR.payload (bytes)
  local mode=$1 r=$2 D=$3 rc=0
  local s e; s=$(date +%s.%N)
  mkdir -p $D/$r.d; cd $D/$r.d
  case $mode in
    singlet_remote) $BIN download $r -o $D/$r.d/$r.1fq --no-profile > log.txt 2>&1 || rc=$? ;;
    singlet_prefetch_local)
      prefetch --max-size u -O sra $r > pf.log 2>&1 && $BIN download sra/$r/$r.sra -o $r.1fq --no-profile > log.txt 2>&1 || rc=$? ;;
    singlet_lossless)
      prefetch --max-size u -O sra $r > pf.log 2>&1 && $BIN download sra/$r/$r.sra -o $r.1fq --no-trim --quality full --no-profile > log.txt 2>&1 || rc=$? ;;
    singlet_orch_io)
      { $BIN download $r --max-reads 1000000 --vdb-threads 2 --no-profile -o head.1fq > head.log 2>&1 &&
        $BIN screen-runs --json head_screen.json head.1fq > screen1.log 2>&1 &&
        prefetch --max-size u -O sra $r > pf.log 2>&1 &&
        $BIN download sra/$r/$r.sra --vdb-threads 1 --no-profile -o $r.1fq > log.txt 2>&1 &&
        $BIN screen-runs --json screen.json $r.1fq > screen2.log 2>&1; } || rc=$? ;;
    fasterq_dump) mkdir -p fq tmp; fasterq-dump --split-files -e 4 -t tmp -O fq $r > log.txt 2>&1 || rc=$? ;;
    prefetch_fasterq_dump)
      mkdir -p fq tmp; prefetch --max-size u -O sra $r > pf.log 2>&1 && fasterq-dump --split-files -e 4 -t tmp -O fq sra/$r/$r.sra > log.txt 2>&1 || rc=$? ;;
    aws_s3) curl -sS --retry 3 -o $r.sra "https://sra-pub-run-odp.s3.amazonaws.com/sra/$r/$r" > log.txt 2>&1 || rc=$? ;;
    ena_https)
      urls=$(curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$r&result=read_run&fields=fastq_ftp&format=tsv" | tail -1 | cut -f2)
      pids=""; for u in ${urls//;/ }; do curl -sS --retry 3 -O "https://$u" >> log.txt 2>&1 & pids="$pids $!"; done
      for p in $pids; do wait $p || rc=$?; done ;;
  esac
  e=$(date +%s.%N)
  echo "$s $e $rc" > $D/$r.t
}

score_fn() {  # MODE D: decode what the mode produced and digest it -> $OUTB/MODE.digest.tsv (srr, file, spots, len, seq, qual)
  local mode=$1 D=$2; : > $OUTB/$mode.digest.tsv
  for r in $RUNS; do
    cd $D/$r.d
    case $mode in
      singlet_*)  # the .1fq holds R1 (barcode + UMI) and R2 (cDNA)
        if [ "$mode" = singlet_orch_io ] || [ -s $r.1fq ]; then
          $BIN decode $r.1fq --r1 $r.R1.fq --r2 $r.R2.fq --threads 4 > decode.log 2>&1
          echo "$r.R1.fq $r.R2.fq" | tr ' ' '\n' | sed "s#^#$D/$r.d/#" ; fi ;;
      fasterq_dump|prefetch_fasterq_dump) ls $D/$r.d/fq/*.fastq 2>/dev/null ;;
      ena_https) ls $D/$r.d/*.fastq.gz 2>/dev/null ;;
    esac
  done | xargs -r -P 8 -I{} $PY -I $SRC/bench/public/harness/download/fq_digest.py {} |
    while IFS=$'\t' read -r f n rl sd qd; do b=$(basename $f); echo -e "${b%%[._]*}\t$b\t$n\t$rl\t$sd\t$qd"; done >> $OUTB/$mode.digest.tsv
  if [ $mode = aws_s3 ]; then
    for r in $RUNS; do echo -e "$r\t$r.sra\t$(stat -c %s $D/$r.d/$r.sra 2>/dev/null)\t0\t$(sha256sum $D/$r.d/$r.sra 2>/dev/null | cut -d' ' -f1)\t-" ; done >> $OUTB/$mode.digest.tsv
  fi
}

for mode in $MODES; do
  D=$SCR/$mode; rm -rf $D; mkdir -p $D
  export HOME=$D/home; mkdir -p $HOME/.ncbi; cp $WORK/cache/condahome/.ncbi/user-settings.mkfg $HOME/.ncbi/
  echo "[dl] $mode start $(date -Is)"
  t0=$(date +%s.%N)
  ( for r in $RUNS; do stream_fn $mode $r $D & done; wait ) &
  grp=$!
  $PY -I $SRC/bench/public/harness/memwatch.py $grp $OUTB/$mode.mem.json &
  mw=$!
  wait $grp; wait $mw
  t1=$(date +%s.%N)
  : > $OUTB/$mode.streams.tsv
  for r in $RUNS; do
    read -r s e rc < $D/$r.t
    pay=$(case $mode in
      singlet_*) stat -c %s $D/$r.d/$r.1fq 2>/dev/null ;;
      aws_s3) stat -c %s $D/$r.d/$r.sra 2>/dev/null ;;
      ena_https) cat $D/$r.d/*.fastq.gz 2>/dev/null | wc -c ;;
      *) du -sbc $D/$r.d/fq 2>/dev/null | tail -1 | cut -f1 ;; esac)
    echo -e "$r\t$s\t$e\t$rc\t${pay:-0}" >> $OUTB/$mode.streams.tsv
  done
  echo -e "ALL\t$t0\t$t1\t0\t0" >> $OUTB/$mode.streams.tsv
  echo "[dl] $mode wall $(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.1f", b-a}') s"; cat $OUTB/$mode.streams.tsv
  if [ "$DIGEST" = 1 ]; then score_fn $mode $D; echo "[dl] $mode scored: $(wc -l < $OUTB/$mode.digest.tsv) files"; fi
  rm -rf $D
done
echo "[dl] done $(date -Is)"

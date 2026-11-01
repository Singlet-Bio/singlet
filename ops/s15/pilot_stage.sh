#!/bin/bash
# S15 beta pilot: one stage of one sample's chain.   pilot_stage.sh STAGE GSM      (Slurm options come from the submitter)
# Layout: $P=$W/pilot/<GSM>/{sra,stage(.1fq),fq,cap1fq,s1cpu,s1gpu,s1cap,starsolo,equiv.txt}. Reads (sra, stage, fq, cap1fq) are deleted by the cleanup stage.
set -uo pipefail
ST=$1; GSM=$2
W=/mnt/projects/debruinz_project/debruinz/singlet; PL=$W/pilot; P=$PL/$GSM; D=$W/jobs/s15
SINGLET_THREADS=16   # the same for cpu, gpu and matched-prefix runs: outputs can depend on thread count (known engine bug, S8)
COMMIT=337b405b; BINDIR=$D/bin/$COMMIT; REFB=/mnt/projects/debruinz_project/cellarium/reference; G=$REFB/GRCh38-2024-A
WL=$W/pipeline-benchmark/whitelists; ENV=$W/envs/pbench; CAPPAIRS=100000000
row=$(awk -F'\t' -v g=$GSM '$1==g' $D/samples.tsv); FIVE=$(echo "$row" | cut -f3); TIER=$(echo "$row" | cut -f5)
export SINGLET_WHITELIST_DIR=$WL PATH=$ENV/bin:$PATH
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR $P; trap 'rm -rf $SCR' EXIT
echo "== s15 $ST $GSM node=$(hostname) $(date -Is)"
assert_bin() {
  [ "$(cut -c1-8 $BINDIR/commit)" = "$COMMIT" ] || { echo "WRONG BINARY COMMIT"; exit 3; }
  ls -l $BINDIR/$1; }
singlet_run() {   # engine outdir [reads-args...]
  eng=$1; out=$2; shift 2; rm -rf $out; mkdir -p $out
  /usr/bin/time -v -o $out/time.txt $BINDIR/singlet-$eng "$@" --ref-base $REFB --out-prefix $out/out/ --threads $SINGLET_THREADS --pipeline \
     --output-format 1pz --engine $eng --memory-tier $TIER --tagged-bam > $out/run.log 2>&1
  rc=$?; echo "$SINGLET_THREADS" > $out/threads.txt
  pt=$(grep -rhoE '"pileup_threads": *[0-9]+' $out/out 2> /dev/null | head -1 | grep -oE '[0-9]+$'); echo "${pt:-unknown}" > $out/pileup_threads.txt
  [ "${pt:-x}" = 1 ] && echo "pileup_threads=1 (stream path)" || { echo "INVALID: pileup_threads=${pt:-unknown} (expected 1, the stream path); whitelists dir present: $(ls -d $BINDIR/../whitelists 2>&1)"; touch $out/INVALID; }
  echo "singlet $eng rc=$rc threads=$SINGLET_THREADS"; grep -E "Elapsed \(wall|Maximum resident" $out/time.txt; return $rc; }
ensure_wl() {   # every .1fq must have been decoded WITH --whitelist-dir (pilot_fetch used to pass only the env var, which singlet download ignores: wl_match_rate=0). Redo from the kept .sra once, under a lock.
  mkdir -p $P/stage
  ( flock 9
    [ -e $P/stage/.wl_ok ] && exit 0
    echo "[ensure_wl] redecoding $GSM with --whitelist-dir"
    for sra in $P/sra/*.sra; do a=$(basename $sra .sra)
      $BINDIR/singlet-cpu download $sra --quality binned --vdb-threads 2 --no-profile --whitelist-dir $WL -o $P/stage/$a.1fq.new 2>&1 | grep -E "singlet-gate|protocol=" | cut -c1-260 >> $P/decode_wl.log
      [ ${PIPESTATUS[0]} = 0 ] || { echo "[ensure_wl] decode failed $a"; exit 9; }
      mv $P/stage/$a.1fq.new $P/stage/$a.1fq; touch $P/stage/$a.1fq.done
    done
    touch $P/stage/.wl_ok; cat $P/decode_wl.log ) 9> $P/.decode.lock
}
reads_arg() { n=$(ls $1/*.1fq | wc -l); if [ $n = 1 ]; then ls $1/*.1fq; else echo "--runs-dir $1"; fi; }
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=no "$@"; }
post_gpu() {
  touch $P/.gpu_ended   # LANE_REST = G:fetch:scpu:starsolo:scap,...  (first = the sample after this one); SELF_* = this sample's job ids
  cid=$(sb --job-name=s15-cleanup-$GSM --partition=short-cpu --cpus-per-task=2 --mem=8G --time=01:00:00 --dependency=afterany:${SELF_C}:${SELF_S}:${SELF_P}         --output=$W/logs/%x-%j.out $D/pilot_stage.sh cleanup $GSM); echo "cleanup job $cid"
  [ -n "${LANE_REST:-}" ] || return 0
  nxt=${LANE_REST%%,*}; rest=; case $LANE_REST in *,*) rest=${LANE_REST#*,};; esac
  IFS=: read g f c s p <<< "$nxt"
  r=$(awk -F'	' -v g=$g '$1==g' $D/samples.tsv); mem=$(echo "$r" | cut -f6); pg=$(echo "$r" | cut -f9); tg=$(echo "$r" | cut -f10)
  printf 'LANE_REST=%s
SELF_C=%s
SELF_S=%s
SELF_P=%s
' "$rest" $c $s $p > $D/lane/$g.env
  sb --job-name=s15-gpu-$g --partition=$pg --gres=gpu:nvidia_h100_nvl:$([ $mem -gt 375 ] && echo 2 || echo 1) --cpus-per-task=16 --mem=${mem}G --time=$tg --dependency=afterany:$f      --output=$W/logs/%x-%j.out $D/pilot_stage.sh sgpu $g; }
case $ST in
fetch)
  assert_bin singlet-cpu
  mkdir -p $D/capdir; [ -f $D/capdir/stage.json ] || echo '{"rate_cap_MBps": 450}' > $D/capdir/stage.json
  python3 -u $D/pilot_fetch.py $GSM $P $D/src $BINDIR/singlet-cpu $WL $D/capdir; rc=$?; du -sh $P/sra $P/stage; exit $rc ;;
fq)   # competitor input: the first CAPPAIRS pairs as FASTQ, plus the same prefix as .1fq for the matched Singlet run
  python3 - <<PY || exit 4
import json, subprocess, os
r = json.load(open("$P/runs.json"))["runs"]; left = $CAPPAIRS; os.makedirs("$P/fq", exist_ok=True)
for acc, n in r:
    if left <= 0: break
    k = min(n, left); left -= k
    subprocess.run(["fastq-dump", "-X", str(k), "--split-files", "--gzip", "-O", "$P/fq", "$P/sra/%s.sra" % acc], check=True)
PY
  ls -l $P/fq | head
  tot=$(python3 -c "import json;print(json.load(open('$P/runs.json'))['pairs'])")
  if [ $tot -gt $CAPPAIRS ]; then mkdir -p $P/cap1fq; a=$(python3 -c "import json;print(json.load(open('$P/runs.json'))['runs'][0][0])")
    $BINDIR/singlet-cpu download $P/sra/$a.sra --quality binned --max-reads $CAPPAIRS --vdb-threads 2 --no-profile --whitelist-dir $WL -o $P/cap1fq/$a.1fq && touch $P/cap1fq/.wl_ok; fi ;;
scpu) assert_bin singlet-cpu; ensure_wl || exit 9; singlet_run cpu $P/s1cpu $(reads_arg $P/stage) ;;
sgpu)   # (ensure_wl below) one of 3 serial GPU lanes: run this sample, then (always) submit its cleanup and the next sample of the lane, so at most 3 GPU jobs exist at once
  . $D/lane/$GSM.env
  trap 'post_gpu; rm -rf $SCR' EXIT
  assert_bin singlet-gpu; nvidia-smi --query-gpu=name --format=csv,noheader
  ls $P/stage/*.done > /dev/null 2>&1 || { echo "no staged reads (fetch failed)"; exit 6; }
  ensure_wl || exit 9
  singlet_run gpu $P/s1gpu $(reads_arg $P/stage) ;;
scap) [ -d $P/cap1fq ] || { echo "full sample <= cap: no matched run needed"; exit 0; }; assert_bin singlet-cpu
  if [ ! -e $P/cap1fq/.wl_ok ]; then a=$(python3 -c "import json;print(json.load(open('$P/runs.json'))['runs'][0][0])"); rm -f $P/cap1fq/*.1fq
    $BINDIR/singlet-cpu download $P/sra/$a.sra --quality binned --max-reads $CAPPAIRS --vdb-threads 2 --no-profile --whitelist-dir $WL -o $P/cap1fq/$a.1fq 2>&1 | grep -E "singlet-gate" | cut -c1-260; touch $P/cap1fq/.wl_ok; fi
  singlet_run cpu $P/s1cap $(reads_arg $P/cap1fq) ;;
equiv) python3 $D/out_equiv.py $P/s1cpu/out $P/s1gpu/out 2>&1 | tee $P/equiv.txt | tail -30 ;;
starsolo)
  cd $P; rm -rf starsolo; mkdir starsolo; cd starsolo
  # barcode read = the 24-30 nt file, cDNA read = the longest file, per run, by the first read's length
  python3 - <<PY > files.env
import gzip, glob, collections
runs = collections.defaultdict(dict)
for f in sorted(glob.glob("$P/fq/*.fastq.gz")):
    acc, n = f.split("/")[-1].replace(".fastq.gz", "").rsplit("_", 1)
    with gzip.open(f, "rt") as h:
        h.readline(); L = len(h.readline().strip())
    runs[acc][n] = (f, L)
bc = []; cd = []
for acc in sorted(runs):
    fl = sorted(runs[acc].values(), key=lambda x: x[1])
    cand = [x for x in fl if 24 <= x[1] <= 30]
    bc.append(cand[0][0]); cd.append(fl[-1][0]); L = cand[0][1]
print("R1=%s\nR2=%s\nBCLEN=%d" % (",".join(bc), ",".join(cd), L))
PY
  cat files.env; . files.env
  # chemistry = what Singlet detected for this sample (protocol tag in its CPU run log; whitelist/UMI per include/singlet/fq/protocol.h); barcode-match autodetect is only the fallback
  TAG=$(grep -ho 'protocol=[A-Za-z0-9._-]*' $P/s1cpu/run.log 2> /dev/null | head -1 | cut -d= -f2); best=""; SRC=singlet
  case $TAG in
    10x-3p-v3|10x-5p-v3|10x-3p-v4|10x-5p-v4) best=3M-february-2018.txt; UMI=12 ;;
    10x-3p-v2|10x-5p-v2) best=737K-august-2016.txt; UMI=10 ;;
    10x-arc-gex) best=gex_737K-arc-v1.txt; UMI=12 ;;
    *) SRC=autodetect; bestf=0
       for w in 737K-august-2016.txt 3M-february-2018.txt; do
         f=$(zcat ${R1%%,*} | awk 'NR%4==2{print substr($0,1,16)}' | head -200000 | grep -cxFf $WL/$w); echo "whitelist $w: $f/200000"
         [ $f -gt $bestf ] && { bestf=$f; best=$w; }
       done
       [ $bestf -gt 20000 ] || { echo "no 10x whitelist matches and Singlet gave no usable protocol tag ($TAG)"; exit 5; }
       UMI=$((BCLEN-16)) ;;
  esac
  STRAND=Forward; case $TAG in *-5p-*) STRAND=Reverse ;; "") [ "$FIVE" = 1 ] && STRAND=Reverse ;; esac
  echo "protocol_tag=$TAG source=$SRC"
  CMD="$W/pipeline-benchmark/bin/STAR-velodump --runThreadN 16 --genomeDir $G/star_2.7.11b --readFilesIn $R2 $R1 --readFilesCommand zcat --soloType CB_UMI_Simple --soloCBwhitelist $WL/$best --soloCBlen 16 --soloUMIstart 17 --soloUMIlen $UMI --soloBarcodeReadLength 0 --soloStrand $STRAND --outSAMtype None --outTmpDir $SCR/startmp --soloFeatures Gene GeneFull Velocyto SJ --soloCellFilter EmptyDrops_CR --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR --clipAdapterType CellRanger4 --outFilterScoreMin 30"
  echo "$CMD" > command.txt; echo "whitelist=$best umi=$UMI strand=$STRAND tag=${TAG:-none} source=$SRC" > chem.txt
  export LD_LIBRARY_PATH=$ENV/lib STAR_VELO_DUMP=$SCR/velo_umis.tsv
  eval "/usr/bin/time -v -o time.txt $CMD" > run.log 2>&1; rc=$?; echo "STARsolo rc=$rc"; grep -E "Elapsed \(wall|Maximum resident" time.txt
  [ -f $SCR/velo_umis.tsv ] && gzip -1 -c $SCR/velo_umis.tsv > velo_umis.tsv.gz
  ls Solo.out/* 2>/dev/null | head; exit $rc ;;
cleanup)   # equiv check, then delete reads and BAMs; keep outputs and logs. A KEEP file in the sample dir defers the deletion (extra tools still to run).
  [ -d $P/s1cpu/out ] && [ -d $P/s1gpu/out ] && python3 $D/out_equiv.py $P/s1cpu/out $P/s1gpu/out > $P/equiv.txt 2>&1
  [ -e $P/KEEP ] && { echo "KEEP present: reads left in place"; exit 0; }
  rm -rf $P/sra $P/stage $P/fq $P/cap1fq
  find $P \( -name '*.bam' -o -name '*.bai' \) -delete
  du -sh $P; ls $P ;;
esac

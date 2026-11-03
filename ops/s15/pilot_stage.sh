#!/bin/bash
# S15 beta pilot: one stage of one sample's chain.   pilot_stage.sh STAGE GSM      (Slurm options come from the submitter)
# Layout: $P=$W/pilot/<GSM>/{sra,stage(.1fq),fq,cap1fq,s1cpu,s1gpu,s1cap,starsolo,equiv.txt}. Reads (sra, stage, fq, cap1fq) are deleted by the cleanup stage.
set -uo pipefail
ST=$1; GSM=$2
W=/mnt/projects/debruinz_project/debruinz/singlet; PL=$W/pilot; P=$PL/$GSM; D=$W/jobs/s15
SINGLET_THREADS=16   # the same for cpu, gpu and matched-prefix runs: outputs can depend on thread count (known engine bug, S8)
COMMIT=337b405b; BINDIR=$D/bin/$COMMIT; REFB=/mnt/projects/debruinz_project/cellarium/reference; G=$REFB/GRCh38-2024-A
WL=$W/pipeline-benchmark/whitelists; ENV=$W/envs/pbench; CAPPAIRS=100000000
row=$(awk -F'\t' -v g=$GSM '$1==g' $D/samples.tsv); FIVE=${FIVE_ENV:-$(echo "$row" | cut -f3)}; TIER=$(echo "$row" | cut -f5)
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
reads_arg() { n=$(ls $1/*.1fq | wc -l); if [ $n = 1 ]; then ls $1/*.1fq; else echo "--runs-dir $1 --runs-used $(ls $1/*.1fq | xargs -n1 basename | sed s/.1fq$// | sort | paste -sd,)"; fi; }
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=no "$@"; }
post_gpu() {
  touch $P/.gpu_ended   # LANE_REST = G:fetch:scpu:starsolo:scap,...  (first = the sample after this one); SELF_* = this sample's job ids
  cid=$(sb --job-name=s15-cleanup-$GSM --partition=short-cpu --cpus-per-task=2 --mem=8G --time=01:00:00 --dependency=afterany:${SELF_C}:${SELF_S}:${SELF_P}         --output=$W/logs/%x-%j.out $D/pilot_stage.sh cleanup $GSM); echo "cleanup job $cid"
  if [ -z "${LANE_REST:-}" ]; then   # end of this lane: take the next sample from the shared extra queue (GPU runs that have to be redone)
    LANE_REST=$( ( flock 8; l=$(head -1 $D/lane/extra.queue 2> /dev/null); [ -n "$l" ] && sed -i 1d $D/lane/extra.queue; echo "$l" ) 8> $D/lane/extra.lock )
  fi
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
  if [ $tot -gt $CAPPAIRS ] && [ -z "${STORE_RUN:-}" ]; then mkdir -p $P/cap1fq; a=$(python3 -c "import json;print(json.load(open('$P/runs.json'))['runs'][0][0])")
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
  # BAMRUN=1 (engine rerun, pilot_rerun.sh rstar): same chemistry as the first STARsolo run, sorted BAM for mgatk/cellsnp/vireo/souporcell, GeneFull only, into starsolo_bam
  SD=starsolo; [ -n "${BAMRUN:-}${STORE_RUN:-}" ] && SD=starsolo_bam
  cd $P; rm -rf $SD; mkdir $SD; cd $SD
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
    if not cand: cand = [(x[0], 28) for x in fl if x[1] > 30][:1]   # barcode read stored untrimmed (e.g. 151 nt): CB/UMI sit in its first 28 nt
    bc.append(cand[0][0]); cd.append(fl[-1][0]); L = cand[0][1]
print("R1=%s\nR2=%s\nBCLEN=%d" % (",".join(bc), ",".join(cd), L))
PY
  SCR_AD=$(mktemp); cat files.env; . files.env
  # chemistry = what Singlet detected for this sample (protocol tag in its CPU run log; whitelist/UMI per include/singlet/fq/protocol.h); barcode-match autodetect is only the fallback
  TAG=$(grep -ho 'protocol=[A-Za-z0-9._-]*' $P/s1cpu/run.log 2> /dev/null | head -1 | cut -d= -f2); best=""; SRC=singlet
  [ -n "${BAMRUN:-}" ] && [ -f $P/starsolo/chem.txt ] && { TAG=$(sed -n 's/.*tag=\([^ ]*\).*//p' $P/starsolo/chem.txt 2> /dev/null | head -1); [ "$TAG" = none ] && TAG=; }
  CBLEN=16
  case $TAG in
    10x-3p-v3|10x-5p-v3|10x-3p-v4|10x-5p-v4) best=3M-february-2018.txt; UMI=12 ;;
    10x-3p-v2|10x-5p-v2) best=737K-august-2016.txt; UMI=10 ;;
    10x-arc-gex) best=gex_737K-arc-v1.txt; UMI=12 ;;
    *) SRC=autodetect; bestf=0; oth=0   # whitelist match on the first 200k reads of the largest R1 file (10x tars have lanes of very different size): needs >50 % and a 3x lead; v1 = 14-nt barcode + 10-nt UMI
       BIG=$(ls -SL ${R1//,/ } | head -1)
       for wl in 737K-august-2016.txt:16 3M-february-2018.txt:16 737K-april-2014.txt:14; do w=${wl%:*}; n=${wl#*:}
         zcat $BIG | awk -v n=$n 'NR%4==2{print substr($0,1,n)}' | head -200000 > $SCR_AD; NR1=$(wc -l < $SCR_AD)
         f=$(grep -cxFf $WL/$w $SCR_AD); echo "whitelist $w (CB $n nt): $f/$NR1 reads ($BIG)"
         if [ $f -gt $bestf ]; then oth=$bestf; bestf=$f; best=$w; CBLEN=$n; bestn=$NR1; elif [ $f -gt $oth ]; then oth=$f; fi
       done
       [ $((bestf * 2)) -gt ${bestn:-1} ] && [ $bestf -gt $((3 * oth)) ] || { echo "no 10x whitelist matches >50 % and Singlet gave no usable protocol tag ($TAG)"; exit 5; }
       UMI=$((BCLEN-CBLEN)) ;;
  esac
  STRAND=Forward; case $TAG in *-5p-*) STRAND=Reverse ;; "") [ "$FIVE" = 1 ] && STRAND=Reverse ;; esac
  echo "protocol_tag=$TAG source=$SRC"
  FEAT="Gene GeneFull Velocyto SJ"; SAMT=None; [ -n "${BAMRUN:-}${STORE_RUN:-}" ] && { FEAT=GeneFull; [ -n "${STORE_RUN:-}" ] && FEAT="Gene GeneFull GeneFull_Ex50pAS Velocyto SJ"; SAMT="BAM SortedByCoordinate --outSAMattributes NH HI nM AS CR UR CB UB GX GN sS sQ sM --limitBAMsortRAM 60000000000"; }
  CMD="$W/pipeline-benchmark/bin/STAR-velodump --runThreadN 16 --genomeDir $G/star_2.7.11b --readFilesIn $R2 $R1 --readFilesCommand zcat --soloType CB_UMI_Simple --soloCBwhitelist $WL/$best --soloCBlen $CBLEN --soloUMIstart $((CBLEN + 1)) --soloUMIlen $UMI --soloBarcodeReadLength 0 --soloStrand $STRAND --outSAMtype $SAMT --outTmpDir $SCR/startmp --soloFeatures $FEAT --soloCellFilter EmptyDrops_CR --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR --clipAdapterType CellRanger4 --outFilterScoreMin 30"
  echo "$CMD" > command.txt; echo "whitelist=$best umi=$UMI strand=$STRAND tag=${TAG:-none} source=$SRC" > chem.txt
  export LD_LIBRARY_PATH=$ENV/lib STAR_VELO_DUMP=$SCR/velo_umis.tsv
  eval "/usr/bin/time -v -o time.txt $CMD" > run.log 2>&1; rc=$?; echo "STARsolo rc=$rc"; grep -E "Elapsed \(wall|Maximum resident" time.txt
  [ -f $SCR/velo_umis.tsv ] && gzip -1 -c $SCR/velo_umis.tsv > velo_umis.tsv.gz
  ls Solo.out/* 2>/dev/null | head
  if [ -n "${STORE_RUN:-}" ]; then ln -sfn starsolo_bam $P/starsolo; [ $rc = 0 ] && python3 $D/store_register.py "${DSID:-$GSM}" "${SMP:-$GSM}" STARsolo "${TVER:-2.7.11b-velodump}" $P/starsolo_bam/Solo.out --cmd-file command.txt --time-file time.txt --input-dir $P; fi
  exit $rc ;;
fetchurl)   # benchmark store: a 10x fastq tar streamed from URL; GEX R1/R2 are linked into fq/ as <stem>_1 / <stem>_2 (the names the starsolo stage expects)
  mkdir -p $P/tar $P/fq; curl -sL --retry 5 --retry-delay 20 "${URL:?URL}" | tar -x -C $P/tar --wildcards '*_R1_*' '*_R2_*' || exit 5
  for f in $(find $P/tar -name '*_R[12]_001.fastq.gz' | grep -viE '_(antibody|adt|vdj|cmo|hto|atac|crispr|cellplex)_S[0-9]+_L|_I[12]_'); do b=$(basename $f); r=$(echo $b | grep -oE '_R[12]_' | tr -d '_R'); ln -sf $f $P/fq/$(echo $b | sed -E 's/_R[12]_001.fastq.gz//')_$r.fastq.gz; done
  echo '{"pairs": 0, "runs": []}' > $P/runs.json; ls $P/fq | head; du -shL $P/fq ;;
fetchbam)   # benchmark store: a submitter's 10x-style BAM (CR/CQ/UR/UQ tags kept) streamed from BAMURL and turned back into reads: R1 = CR+UR (barcode+UMI) and R2 = the cDNA read, primary alignments only
  mkdir -p $P/fq; set -o pipefail; ACC=${ACC:?ACC}
  curl -s --retry 5 --retry-delay 20 "${BAMURL:?BAMURL}" | $W/envs/pbench/bin/samtools fastq -@ 4 -F 2304 -T CR,CQ,UR,UQ - 2> $P/fetchbam.err | awk -F'\t' -v r1="gzip -1 > $P/fq/${ACC}_1.fastq.gz" -v r2="gzip -1 > $P/fq/${ACC}_2.fastq.gz" '
    NR%4==1 {nm=$1; cr="";cq="";ur="";uq=""; for(i=2;i<=NF;i++){t=substr($i,1,2); v=substr($i,6); if(t=="CR")cr=v; else if(t=="CQ")cq=v; else if(t=="UR")ur=v; else if(t=="UQ")uq=v}; next}
    NR%4==2 {sq=$0; next}
    NR%4==0 {if(cr!="" && ur!="" && cq!="" && uq!=""){n++; print nm"\n"cr ur"\n+\n"cq uq | r1; print nm"\n"sq"\n+\n"$0 | r2}}
    END {print n+0 > "/dev/stderr"}' 2> $P/fetchbam.count || exit 5
  N=$(tail -1 $P/fetchbam.count); echo "{\"pairs\": $N, \"runs\": [[\"$ACC\", $N]]}" > $P/runs.json; ls -l $P/fq; echo "reads kept: $N" ;;
storeclean) rm -rf $P/sra $P/fq $P/tar $P/stage $P/cap1fq; find $P \( -name '*.bam' -o -name '*.bai' \) -delete; du -sh $P ;;
cleanup)   # equiv check, then delete reads and BAMs; keep outputs and logs. A KEEP file in the sample dir defers the deletion (extra tools still to run).
  [ -d $P/s1cpu/out ] && [ -d $P/s1gpu/out ] && python3 $D/out_equiv.py $P/s1cpu/out $P/s1gpu/out > $P/equiv.txt 2>&1
  [ -e $P/KEEP ] && { echo "KEEP present: reads left in place"; exit 0; }
  rm -rf $P/stage $P/fq $P/cap1fq   # the .sra files stay: Singlet is rerun from them on the fixed engine; they are deleted after those reruns
  find $P \( -name '*.bam' -o -name '*.bai' \) -delete
  du -sh $P; ls $P ;;
esac

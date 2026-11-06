#!/bin/bash
#SBATCH --job-name=ghc-prof
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=250G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# gpu/bold-hostcut: host profile of ONE --engine gpu run of pbmc1k_v3 (T=7, 2 slots, level 5 + ingest, as the timing job) per pipeline stage.
# Stages come from the run log: refs = start .. "STAR pid=" (references, .1fq scan and decode, R11 histogram, genome preload),
# stream = "STAR pid=" .. "[gpu_pileup] stream drained" (STAR prefill + map + stream into the pileup), tail = the rest.
# Method 1: `perf record -g` with the realtime clock (when perf runs here: node PATH, then the pbench env). perf report is cut per stage with --time.
# Method 2 (always, and the fallback when perf is missing or refused): a 0.25 s sampler of /proc/<pid>/task/*/stat grouped by process and thread name,
# with the cgroup core-s per stage.
# Parameters (--export): BINDIR + EXPECT_COMMIT (build with symbols: EXTRA_CXX="-g -fno-omit-frame-pointer"; the plain build also works, with fewer call frames),
# SAMPLE (pbmc1k_v3), T (7), SLOTS (2), FREQ (149). ALLOW_NO_EMBED=1 is accepted.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:?}; BIN=$BINDIR/singlet-gpu; EXPECT_COMMIT=${EXPECT_COMMIT:?}
SAMPLE=${SAMPLE:-pbmc1k_v3}; T=${T:-7}; SLOTS=${SLOTS:-2}; FREQ=${FREQ:-149}
COMMIT=$(cat $BINDIR/commit-gpu); case $COMMIT in $EXPECT_COMMIT*) ;; *) echo "REFUSED: commit $COMMIT"; exit 2 ;; esac
EMBED=no; grep -aqF "$COMMIT" $BIN && EMBED=yes
[ $EMBED = yes ] || [ "${ALLOW_NO_EMBED:-0}" = 1 ] || { echo "REFUSED: commit $COMMIT is not inside the binary"; exit 2; }
R=$W/jobs/gpu-hostcut/results/prof-$SLURM_JOB_ID; mkdir -p $R; S=$R/summary.txt
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) bin=$BIN sha256=$(sha256sum $BIN | cut -d' ' -f1) commit=$COMMIT sample=$SAMPLE T=$T slots=$SLOTS $(date -Is)" | tee $S
SCR=/tmp/$USER-ghcpf-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'kill $(jobs -p) 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_SLOTS=$SLOTS
case $SAMPLE in neuron1k_v3|nuclei2k_v2) G=$REF/GRCm39-2024-A ;; hgmm1k_v3) G=$REF/GRCh38-GRCm39-barnyard ;; *) G=$REF/GRCh38-2024-A ;; esac
src=$PB/data/$SAMPLE/$SAMPLE.1fq; [ -f $src ] || src=$W/jobs/fable/release/data/$SAMPLE/$SAMPLE.1fq
cp $src $SCR/$SAMPLE.1fq
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')

# --- which profiler can run here ----------------------------------------------------------------------------------------------------
PERF=$(command -v perf || true)
echo "perf: ${PERF:-none on PATH}; perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null); kptr_restrict=$(cat /proc/sys/kernel/kptr_restrict 2>/dev/null)" | tee -a $S
PERFARGS=""
if [ -n "$PERF" ]; then
  for ev in "-e cycles" "-e cycles --all-user" "-e cpu-clock" "-e cpu-clock --all-user"; do
    if $PERF record $ev -F 99 -o $SCR/probe.data -- /bin/true > $SCR/probe.log 2>&1 && [ -s $SCR/probe.data ]; then PERFARGS="$ev"; break; fi
  done
  echo "perf record arguments that work: ${PERFARGS:-none (perf is refused here: $(head -2 $SCR/probe.log | tr '\n' ' '))}" | tee -a $S
fi

# --- the run, with the thread sampler and the cgroup meter ---------------------------------------------------------------------------
python3 - "$R/threads.tsv" <<'PYEOF' &
import os, sys, time
out = open(sys.argv[1], "w"); out.write("t\tpid\tpcomm\ttid\ttcomm\tcpu_ticks\n")
me = os.getpid(); last = {}
tick = os.sysconf("SC_CLK_TCK")
def walk():
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me: continue
        try:
            with open(f"/proc/{pid}/comm") as f: pc = f.read().strip()
            for tid in os.listdir(f"/proc/{pid}/task"):
                try:
                    with open(f"/proc/{pid}/task/{tid}/stat") as f: st = f.read()
                    r = st.rindex(")"); name = st[st.index("(") + 1:r]; fl = st[r + 2:].split()
                    yield pid, pc, tid, name, int(fl[11]) + int(fl[12])   # utime + stime
                except Exception: pass
        except Exception: pass
while True:
    t = time.time()
    for pid, pc, tid, name, tk in walk():
        if not (pc.startswith("singlet") or pc.startswith("STAR") or pc.startswith("perf")): continue
        key = (pid, tid)
        if last.get(key) != tk:
            out.write(f"{t:.2f}\t{pid}\t{pc}\t{tid}\t{name}\t{tk}\n"); last[key] = tk
    out.flush(); time.sleep(0.25)
PYEOF
SAMP=$!
( echo -e "t\tcpu_s"; while true; do printf "%s\t%s\n" $(date +%s.%N) $(awk '$1=="usage_usec"{printf "%.2f", $2/1e6}' $CG/cpu.stat); sleep 1; done ) > $R/monitor.tsv 2>/dev/null &
MON=$!
cd $SCR
RUNCMD=($BIN $SCR/$SAMPLE.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads $T --pipeline --output-format 1pz --engine gpu)
T0=$(date +%s.%N)
if [ -n "$PERFARGS" ]; then
  /usr/bin/time -v -o $R/time.txt $PERF record $PERFARGS -F $FREQ -g -k CLOCK_REALTIME -o $R/perf.data -- "${RUNCMD[@]}" 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $R/run.log.ts
else
  /usr/bin/time -v -o $R/time.txt "${RUNCMD[@]}" 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $R/run.log.ts
fi
T1=$(date +%s.%N); kill $SAMP $MON 2>/dev/null
echo "run: wall $(echo "$T1 - $T0" | bc) s; $(grep -a 'Maximum resident' $R/time.txt | sed 's/^[ \t]*//')" | tee -a $S

# --- stage windows from the log ----------------------------------------------------------------------------------------------------
tsof() { grep -a -m1 -F "$1" $R/run.log.ts | cut -d' ' -f1; }
L0=$(head -1 $R/run.log.ts | cut -d' ' -f1)
A=$(tsof "STAR pid="); B=$(tsof "[gpu_pileup] stream drained"); C=$(tsof "STAR completed")
[ -z "$B" ] && B=$C
L1=$(tail -1 $R/run.log.ts | cut -d' ' -f1)
echo "stage windows (epoch s): start=$L0 STAR_pid=$A stream_drained=$B STAR_completed=$C end=$L1" | tee -a $S
printf "stage\tstart\tend\n" > $R/windows.tsv
printf "refs\t%s\t%s\nstream\t%s\t%s\ntail\t%s\t%s\n" $L0 $A $A $B $B $L1 >> $R/windows.tsv
# cgroup core-s per stage
python3 - $R <<'PYEOF' | tee -a $S
import sys
R = sys.argv[1]
mon = [l.split("\t") for l in open(f"{R}/monitor.tsv").read().splitlines()[1:] if l.count("\t") == 1 and l.split("\t")[1]]
mon = [(float(a), float(b)) for a, b in mon]
def at(t):
    for i in range(1, len(mon)):
        if mon[i][0] >= t:
            a, b = mon[i - 1], mon[i]; return a[1] + (b[1] - a[1]) * (t - a[0]) / max(b[0] - a[0], 1e-9)
    return mon[-1][1]
print("%-8s %8s %9s" % ("stage", "wall s", "core-s"))
for l in open(f"{R}/windows.tsv").read().splitlines()[1:]:
    n, a, b = l.split("\t"); a, b = float(a), float(b)
    print("%-8s %8.0f %9.1f" % (n, b - a, at(b + 1) - at(a)))   # +1: the log stamps are whole seconds
PYEOF

# --- method 1: perf per stage -----------------------------------------------------------------------------------------------------
if [ -n "$PERFARGS" ] && [ -s $R/perf.data ]; then
  ls -l $R/perf.data | tee -a $S
  while IFS=$'\t' read -r name a b; do
    [ "$name" = stage ] && continue
    [ -n "$a" ] && [ -n "$b" ] || continue
    for srt in "comm,sym" "dso"; do
      f=$R/perf-$name-${srt//,/_}.txt
      $PERF report -i $R/perf.data --time $a.0,$b.999 --no-children --sort $srt -g none --percent-limit 0.4 --stdio > $f 2> $f.err
    done
    # the callers of the top symbols: caller-ordered report
    $PERF report -i $R/perf.data --time $a.0,$b.999 --children --sort sym -g caller,0.5,callee --percent-limit 4 --stdio 2>/dev/null | head -400 > $R/perf-$name-callers.txt
    # copy and decode shares (heuristic by symbol name): memcpy and friends, page clearing and faults, read()/decompress/decode
    awk -v st=$name '/^ *[0-9.]+%/ { p=$1; sub("%","",p); sym=$0; tot+=p;
         if (sym ~ /memcpy|memmove|__copy|copy_user|copy_page|rep_movs|_copy_to|_copy_from|clear_page|page_fault|handle_mm_fault|do_anonymous|migrate_pages|compact/) cp+=p;
         else if (sym ~ /read|pread|filemap|generic_file|ext4_|lustre|nfs|zstd|ZSTD|inflate|decompress|decode|Decode|unpack|pz|1fq|FQ|ingest|Ingest/) dc+=p }
         END { printf "%s: listed %.1f%%, copy/fault-like %.1f%%, read/decode-like %.1f%%\n", st, tot, cp, dc }' $R/perf-$name-comm_sym.txt | tee -a $S
  done < $R/windows.tsv
  echo "=== top symbols per stage (comm, symbol; percent of all samples of the stage window)" | tee -a $S
  for name in refs stream; do echo "--- $name" | tee -a $S; grep -a -E '^ *[0-9.]+%' $R/perf-$name-comm_sym.txt | head -25 | tee -a $S; done
else
  echo "no perf data: the thread sampler below is the profile" | tee -a $S
fi

# --- method 2: thread sampler per stage ------------------------------------------------------------------------------------------------
python3 - $R <<'PYEOF' | tee -a $S
import sys, collections
R = sys.argv[1]; tick = 100.0
win = [l.split("\t") for l in open(f"{R}/windows.tsv").read().splitlines()[1:]]
rows = []
for l in open(f"{R}/threads.tsv").read().splitlines()[1:]:
    t, pid, pc, tid, tc, tk = l.split("\t"); rows.append((float(t), pid, pc, tid, tc, int(tk)))
rows.sort()
for name, a, b in win:
    a, b = float(a), float(b) + 1
    last = {}; agg = collections.Counter(); cnt = collections.Counter()
    for t, pid, pc, tid, tc, tk in rows:
        key = (pid, tid)
        if t < a:
            last[key] = tk; continue
        if t > b: continue
        d = tk - last.get(key, tk); last[key] = tk
        # thread names carry a trailing index in some pools: strip digits
        import re
        k = (pc, re.sub(r"[-_ ]?\d+$", "", tc) or tc)
        agg[k] += d; cnt[k] += 0
    tot = sum(agg.values()) / tick
    print(f"=== thread sampler, stage {name}: {tot:.0f} core-s (ticks)")
    for (pc, tc), v in agg.most_common(12):
        print(f"  {v / tick:8.1f} core-s  {100 * v / max(sum(agg.values()), 1):5.1f}%  {pc} / {tc}")
PYEOF

# --- what the log itself says about refs -------------------------------------------------------------------------------------------
echo "=== log lines inside the refs window (whole-second stamps, offset from start)" | tee -a $S
awk -v t0=$L0 -v a=$A '$1 <= a && ($0 ~ /Genome preload|decoded into the packed|R11 exact|References loaded|references loaded|GPU probe|Scanned|CR4|nonhost|Loading|barcode universe/) { printf "+%ds %s\n", $1 - t0, substr($0, 12, 200) }' $R/run.log.ts | tee -a $S
echo done $(date -Is) | tee -a $S

# Shared by the release-gate jobs (sourced). One pinned binary, whitelists next to it, stream-path assertion.
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; RG=$W/release-gate
REF=/mnt/projects/debruinz_project/cellarium/reference
PIN8=${PIN8:-d9834b8a}
ENGINE=${ENGINE:-cpu}                      # cpu: singlet-cpu (engine off); gpu: singlet-gpu --engine gpu
BIN=$RG/bin/$PIN8/singlet-$ENGINE
ENGINE_ARGS=(); [ "$ENGINE" = gpu ] && ENGINE_ARGS=(--engine gpu)
rg_check_bin() {
  [ -x "$BIN" ] || { echo "[rg] missing $BIN"; exit 3; }
  [ -d "$(dirname "$BIN")/../whitelists" ] || { echo "[rg] whitelists must sit at $(dirname "$BIN")/../whitelists (else the sorted path is taken silently)"; exit 3; }
  echo "[rg] binary $BIN sha256=$(sha256sum "$BIN" | cut -c1-16) commit=$(cat $(dirname "$BIN")/commit) engine=$ENGINE"
}
# Assert the unsorted stream path ran: the engine logs it on the stream path and never when it fell back to the sorted BAM.
rg_stream_ok() {   # $1 = run.log
  if grep -q "BAM: unsorted STAR stream into the pileup" "$1"; then
    echo "STREAM PATH OK"
  else
    echo "STREAM PATH FAIL: $1 lacks the unsorted-stream line (whitelist missing -> sorted path?)"; return 1
  fi
}

#!/usr/bin/env bash
# usage: tools/idea2_final.sh BIN logfile outtsv "ENV=.. ENV=.." [reps] [codec] [files...]
# runs the given codec over the corpus (exon + intron of every sample); decode verified when reps>0
cd "$(dirname "$0")/.."
bin="$1"; log="$2"; out="$3"; envs="$4"; reps="${5:-1}"; codec="${6:-idea2_env}"; shift 6 2>/dev/null
files="$@"
if [ -z "$files" ]; then
  files=""
  for s in tumor_3p pbmc_v2_typical pbmc_v3_lowgenes pbmc_v2_shallow brain_sn_v3_intronic tcell_5p brain_sn_v3_deep; do
    files="$files corpus/$s/exon_counts.1pz corpus/$s/intron_counts.1pz"
  done
fi
for f in $files; do
  env $envs ./bin/${bin}.exe --no-file-check --reps $reps --codec $codec --out $out $f >> $log 2>&1
done
echo "FINAL-DONE" >> $log

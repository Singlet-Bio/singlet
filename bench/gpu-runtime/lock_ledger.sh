O=$1
for d in 0 1; do for j in $(seq 0 11); do
  L=$O/pool$d/$j.log; E=$O/engine.c$d.$j.log; [ -s $L ] || continue
  t0=$(awk -F'\t' -v j=$j 'NR>1&&$1==j{printf "%d", $5}' $O/pool$d/samples.tsv); wall=$(awk -F'\t' -v j=$j 'NR>1&&$1==j{printf "%.0f", $7}' $O/pool$d/samples.tsv)
  ts() { grep -a -m1 -E "$1" $L | awk -v t0=$t0 '{print $1-t0}'; }
  mw=$(grep -a -m1 -E "ENGINE_GPU: device [0-9]" $E | sed -E 's/.*waited ([0-9.e+-]+) s for it, held it ([0-9.e+-]+) s.*/\1 \2/')
  sw=$(grep -a -m1 "E11: Solo on the device" $E | sed -E 's/.*waited ([0-9.e+-]+) s for the device.*/\1/')
  pl=$(grep -a -m1 "gpu_pileup\] device lock" $L | sed -E 's/.*waited ([0-9.e+-]+) s, held ([0-9.e+-]+) s.*/\1 \2/')
  nh=$(grep -a -m1 "nonhost\] engine_gpu: device" $L | sed -E 's/.*\(waited ([0-9.e+-]+) s\).*/\1/')
  echo "c$d.$j wall=$wall scan=$(ts 'decoded into the packed') handoff=$(ts 'hand-off pass: [0-9]') pileup_done=$(ts 'step run_pipeline') allsite=$(ts 'allsite-gpu\]') cellcall=$(ts 'cell_calling\] n_cells') export_end=$(ts 'export\] Total') | map wait,held=${mw:-?} solo_wait=${sw:-?} pileup wait,held=${pl:-?} nonhost_wait=${nh:-?}"
done; done

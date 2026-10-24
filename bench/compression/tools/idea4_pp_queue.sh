#!/usr/bin/env bash
# sequentially build the model sets for the remaining folds (after the primary one finished)
cd "$(dirname "$0")/.."
export OMP_WAIT_POLICY=PASSIVE
build() { # fold ks iters
  LINE=$(grep "^$1|" tools/idea4_pp_folds.txt); TRAIN=$(echo "$LINE" | cut -d'|' -f3)
  need=""
  for k in $(echo $2 | tr ',' ' '); do [ -f results/idea4_pp_model_$1_k${k}.bin ] || need="$need,$k"; done
  need=${need#,}
  [ -z "$need" ] && return
  echo "[queue] $1 ks=$need"; python -P tools/idea4_pp_build.py --train "$TRAIN" --name $1 --ks $need --cap 2500 --iters $3
}
while pgrep -f "idea4_pp_build.py --train h_pbmc_a,h_pbmc_b,h_b,h_c,h_d,pbmc_v3_lowgenes,tcell_5p,tumor_3p --name F142444" > /dev/null; do sleep 10; done
build M285936 0,16,32 30
build F244263 0,16 30
build F152522 0,16 30
build F145370 0,16 30
build XH 0,16 30
build IShuman 0,16 30
build ISmouse 0,16 30
build F244263 32 30
build F152522 32 30
build F145370 32 30
build XH 32 30
build IShuman 32 30
build M285936 8 30
echo "[queue] done"

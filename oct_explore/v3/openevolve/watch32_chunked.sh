#!/bin/bash
cd /workspace/qumode_sep/oct_explore/v3/openevolve
while pgrep -f "openevolve.cli runs/net32o" >/dev/null; do
  for c in runs/net32o/checkpoints/checkpoint_*; do k=$(basename $c); [ -f $c/best_program.py ] || continue
    [ -s runs/net32o/check/${k}_chunked.json ] && continue
    MEMMB=3500 NEED=6700 ./run_paused.sh runs/net32o/check/${k}_chunked.json OE_TRAIN_N=10,15 OE_VAL_N=15 OE_CHECK_N=20 OE_NVAL=30 OE_BUDGET=150 taskset -c 0-7 python3.13 net/chunked_check.py $c/best_program.py
    echo "$(date +%T) $k: $(python3.13 -c "import json;d=json.load(open('runs/net32o/check/${k}_chunked.json'));print(d.get('n20_single_median'),d.get('n20_single_B90'),d.get('n20_rr_median'),d.get('n20_rr_B90'),d.get('error'))")"
  done; sleep 120
done

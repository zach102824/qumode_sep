#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_NUM_INTRA_OP_THREADS=1 JAX_NUM_INTER_OP_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
SCRIPT="/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py"
LOGDIR="/workspace/qumode_sep/oct_explore/logs/oct23_exp"
RESDIR="/workspace/qumode_sep/oct_explore/results/oct23_exp"
# Only tags that current wave2 already passed over without producing JSON
MISSING=(
  "--tag gnn5_n6_1p4x1_bpdec_pcap --n 6 --sched 1p4x1 --order csd --relabel bpdec --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 3000 --bs 250 --lr 0.0012 --p_cap 0.65 --tailw 2.5 --init_base $RESDIR/n6_1p4x1_msdec_sdir.json"
)
for args in "${MISSING[@]}"; do
  tag=$(echo "$args" | awk '{for(i=1;i<=NF;i++) if($i=="--tag") print $(i+1)}')
  if [ -f "$RESDIR/${tag}.json" ]; then
    echo "Skipping $tag (already done)"
    continue
  fi
  # Wait until fewer than 6 part1/gnn workers so we leave headroom
  while true; do
    n=$(pgrep -fc 'part1_2cav.py|gnn_5block.py' || true)
    if [ "${n:-0}" -lt 6 ]; then break; fi
    echo "$(date +%H:%M:%S) waiting for slot ($n workers)..."
    sleep 30
  done
  echo "Starting $tag ..."
  python3 "$SCRIPT" $args > "$LOGDIR/${tag}.log" 2>&1
  echo "Finished $tag"
  # refresh report if present
  python3 /workspace/qumode_sep/oct_explore/oct23_exp/build_report.py >/dev/null 2>&1 || true
done
echo "REQUEUE DONE"

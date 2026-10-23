#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_NUM_INTRA_OP_THREADS=1 JAX_NUM_INTER_OP_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"

SCRIPT="/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py"
REP="/workspace/qumode_sep/oct_explore/oct23_exp/build_part2_report.py"
LOGDIR="/workspace/qumode_sep/oct_explore/logs/oct23_exp_part2"
RESDIR="/workspace/qumode_sep/oct_explore/results/oct23_exp_part2"
mkdir -p "$LOGDIR" "$RESDIR"

# WAVE4: (A) flatten gray-GNN B90 slope; (B) push physical SNAP+D GNN toward WS tries
WAVE4_JOBS=(
  "--tag p2_gnn_gray_L15_mp128_cw --mode gnn --mixer gray --L 15 --pairs 1 --kprep 2 --arch mp128 --clause_w --steps 800 --init_shared $RESDIR/p2_gray_L15.json"
  "--tag p2_gnn_gray_L15_attn_cw --mode gnn --mixer gray --L 15 --pairs 1 --kprep 2 --arch attn --clause_w --steps 800 --init_shared $RESDIR/p2_gray_L15.json"
  "--tag p2_gnn_gray_L15_mp64_cw_s1200 --mode gnn --mixer gray --L 15 --pairs 1 --kprep 2 --arch mp64 --clause_w --steps 1200 --init_shared $RESDIR/p2_gray_L15.json"
  "--tag p2_gnn_L15_k3_mp64_cw_ntr10k --mode gnn --L 15 --pairs 3 --kprep 3 --arch mp64 --clause_w --steps 800 --init_shared $RESDIR/p2_shared_L15_k3_ntr10000.json"
  "--tag p2_gnn_L15_k3_mp64_nocw_ntr10k --mode gnn --L 15 --pairs 3 --kprep 3 --arch mp64 --steps 800 --init_shared $RESDIR/p2_shared_L15_k3_ntr10000.json"
  "--tag p2_gnn_L15_k3_mp128_cw_s1200 --mode gnn --L 15 --pairs 3 --kprep 3 --arch mp128 --clause_w --steps 1200 --init_shared $RESDIR/p2_shared_L15_k3.json"
  "--tag p2_gnn_L12_k3_mp64_cw --mode gnn --L 12 --pairs 3 --kprep 3 --arch mp64 --clause_w --steps 800 --init_shared $RESDIR/p2_shared_L15_k3.json"
  "--tag p2_gnn_gray_L15_ds_cw --mode gnn --mixer gray --L 15 --pairs 1 --kprep 2 --arch ds --clause_w --steps 800 --init_shared $RESDIR/p2_gray_L15.json"
)

run_one() {
  local args="$1"
  local tag
  tag=$(echo "$args" | awk '{for(i=1;i<=NF;i++) if($i=="--tag") print $(i+1)}')
  if [ -f "$RESDIR/${tag}.json" ]; then
    echo "Skipping $tag (already done)"
    return 0
  fi
  echo "Starting $tag ..."
  taskset -c 0-5 python3 "$SCRIPT" $args > "$LOGDIR/${tag}.log" 2>&1
  python3 "$REP" >/dev/null 2>&1 || true
  echo "Finished $tag"
}
export -f run_one
export SCRIPT REP LOGDIR RESDIR

printf "%s\n" "${WAVE4_JOBS[@]}" | xargs -I {} -P 6 bash -c 'run_one "$@"' _ {}
echo "ALL WAVE4 JOBS COMPLETE"

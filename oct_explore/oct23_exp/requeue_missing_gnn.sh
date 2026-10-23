#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_NUM_INTRA_OP_THREADS=1 JAX_NUM_INTER_OP_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"

P1_SCRIPT="/workspace/qumode_sep/oct_explore/oct23_exp/part1_2cav.py"
GNN_SCRIPT="/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py"
LOGDIR="/workspace/qumode_sep/oct_explore/logs/oct23_exp"
RESDIR="/workspace/qumode_sep/oct_explore/results/oct23_exp"

JOBS=(
  # 1. Fast n=6 ECD/ECDT none runs and mpw4 1p4x1 shared runs
  "P1 --tag n6_0p5x1_ecd_none  --n 6 --sched 0p5x1 --order csd --gate ecd  --ntrain 2000 --relabel none"
  "P1 --tag n6_1p4x1_ecd_none  --n 6 --sched 1p4x1 --order csd --gate ecd  --ntrain 2000 --relabel none"
  "P1 --tag n6_1p4x1_ecdt_none --n 6 --sched 1p4x1 --order csd --gate ecdt --ntrain 2000 --relabel none"
  "P1 --tag n6_1p4x1_csd_mpw4  --n 6 --sched 1p4x1 --order csd --gate sd   --ntrain 10000 --nscreen 1000 --nstarts 8 --screen_it 50 --refine_top 1 --maxiter 100 --relabel mpw4 --asym"
  "P1 --tag n8_1p4x1_csd_mpw4  --n 8 --sched 1p4x1 --order csd --gate sd   --ntrain 10000 --nscreen 1000 --nstarts 8 --screen_it 50 --refine_top 1 --maxiter 100 --relabel mpw4 --asym"

  # 2. Category C: Pure GNN from raw factor-graph degrees (fv[:,:,3:7]=0, --pure_gnn --relabel none), both --no_clause_w and --clause_w
  "GNN --tag gnn5_n6_pure_nocw --n 6 --sched 1p4x1 --order csd --relabel none --pure_gnn            --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam_bfgs --steps 1200 --maxiter 60 --bs 250 --lr 0.0012 --tailw 1.0 --init_base $RESDIR/n6_ntrain10000_none.json"
  "GNN --tag gnn5_n6_pure_cw   --n 6 --sched 1p4x1 --order csd --relabel none --pure_gnn --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam_bfgs --steps 1200 --maxiter 60 --bs 250 --lr 0.0012 --tailw 1.0 --init_base $RESDIR/n6_ntrain10000_none.json"
  "GNN --tag gnn5_n8_pure_nocw --n 8 --sched 1p4x1 --order csd --relabel none --pure_gnn            --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam_bfgs --steps 1200 --maxiter 60 --bs 250 --lr 0.0012 --tailw 1.0 --init_base $RESDIR/n8_ntrain10000_none.json"
  "GNN --tag gnn5_n8_pure_cw   --n 8 --sched 1p4x1 --order csd --relabel none --pure_gnn --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam_bfgs --steps 1200 --maxiter 60 --bs 250 --lr 0.0012 --tailw 1.0 --init_base $RESDIR/n8_ntrain10000_none.json"

  # 3. Category D: BP/Decimation-Assisted GNN with Fixed Global Cost (--no_clause_w) and mpw4
  "GNN --tag gnn5_n6_1p4x1_bpdec_nocw  --n 6 --sched 1p4x1 --order csd --relabel bpdec            --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 1400 --bs 250 --lr 0.0012 --p_cap 0.65 --tailw 2.5 --init_base $RESDIR/n6_1p4x1_msdec_sdir.json"
  "GNN --tag gnn5_n8_2p3x1_bpdec_nocw  --n 8 --sched 2p3x1 --order csd --relabel bpdec            --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 1400 --bs 250 --lr 0.0012 --p_cap 0.60 --tailw 2.5"
  "GNN --tag gnn5_n8_0p5x1_sdc_mpw4_cw --n 8 --sched 0p5x1 --order sdc --relabel mpw4  --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 1400 --bs 250 --lr 0.0012 --p_cap 0.65 --tailw 2.5 --init_base $RESDIR/n8_0p5x1_sdc_mpw4.json"
)

run_job() {
  local spec="$1"
  local kind=$(echo "$spec" | awk '{print $1}')
  local args=$(echo "$spec" | cut -d' ' -f2-)
  local tag=$(echo "$args" | awk '{for(i=1;i<=NF;i++) if($i=="--tag") print $(i+1)}')
  if [ -f "$RESDIR/${tag}.json" ]; then
    echo "Skipping $tag (already done)"
    return 0
  fi
  while [ "$(pgrep -fc 'part1_2cav.py|gnn_5block.py' || true)" -ge 6 ]; do
    sleep 5
  done
  echo "Starting $tag ..."
  if [ "$kind" = "P1" ]; then
    taskset -c 0-5 python3 "$P1_SCRIPT" $args > "$LOGDIR/${tag}.log" 2>&1
  else
    taskset -c 0-5 python3 "$GNN_SCRIPT" $args > "$LOGDIR/${tag}.log" 2>&1
  fi
  echo "Finished $tag"
  python3 /workspace/qumode_sep/oct_explore/oct23_exp/build_report.py >/dev/null 2>&1 || true
}
export -f run_job
export P1_SCRIPT GNN_SCRIPT LOGDIR RESDIR

printf "%s\n" "${JOBS[@]}" | xargs -I {} -P 2 bash -c 'run_job "$@"' _ {}
echo "ALL REQUEUE JOBS COMPLETE"

#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_NUM_INTRA_OP_THREADS=1 JAX_NUM_INTER_OP_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"

P1="/workspace/qumode_sep/oct_explore/oct23_exp/part1_2cav.py"
GNN="/workspace/qumode_sep/oct_explore/oct23_exp/gnn_5block.py"
AUD="/workspace/qumode_sep/oct_explore/oct23_exp/audit_one.py"
LOGDIR="/workspace/qumode_sep/oct_explore/logs/oct23_exp"
RESDIR="/workspace/qumode_sep/oct_explore/results/oct23_exp"

JOBS=(
  # 1. Clean post-fix n=6 bpdec N_train curve (500, 2000, 5000)
  "P1 --tag n6_ntrain500_bpdec  --n 6 --sched 1p4x1 --order csd --gate sd --ntrain 500  --nscreen 500  --nstarts 8 --screen_it 60 --refine_top 1 --maxiter 120 --relabel bpdec --asym"
  "P1 --tag n6_ntrain2000_bpdec --n 6 --sched 1p4x1 --order csd --gate sd --ntrain 2000 --nscreen 1000 --nstarts 8 --screen_it 60 --refine_top 1 --maxiter 120 --relabel bpdec --asym"
  "P1 --tag n6_ntrain5000_bpdec --n 6 --sched 1p4x1 --order csd --gate sd --ntrain 5000 --nscreen 1000 --nstarts 8 --screen_it 60 --refine_top 1 --maxiter 120 --relabel bpdec --asym"

  # 2. Category C: Pure GNN (--pure_gnn --relabel none, fv[:,:,3:7]=0) on 0p5x1 (sdc) and 2p3x1 (csd) at n=6 and n=8
  "GNN --tag gnn5_n6_0p5x1_sdc_pure_nocw --n 6 --sched 0p5x1 --order sdc --relabel none --pure_gnn            --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 800 --bs 250 --lr 0.0012 --tailw 1.0 --init_base $RESDIR/n6_0p5x1_sdc_none.json"
  "GNN --tag gnn5_n6_0p5x1_sdc_pure_cw   --n 6 --sched 0p5x1 --order sdc --relabel none --pure_gnn --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 800 --bs 250 --lr 0.0012 --tailw 1.0 --init_base $RESDIR/n6_0p5x1_sdc_none.json"
  "GNN --tag gnn5_n8_2p3x1_pure_nocw     --n 8 --sched 2p3x1 --order csd --relabel none --pure_gnn            --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 800 --bs 250 --lr 0.0012 --tailw 1.0 --init_base $RESDIR/n8_2p3x1_csd_none.json"
  "GNN --tag gnn5_n8_2p3x1_pure_cw       --n 8 --sched 2p3x1 --order csd --relabel none --pure_gnn --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 800 --bs 250 --lr 0.0012 --tailw 1.0 --init_base $RESDIR/n8_2p3x1_csd_none.json"

  # 3. Category D: Clean post-fix BP/Decimation-Assisted GNN runs (both fixed cost and clause_w)
  "GNN --tag gnn5_n6_1p4x1_bpdec_nocw    --n 6 --sched 1p4x1 --order csd --relabel bpdec            --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 800 --bs 250 --lr 0.0012 --p_cap 0.65 --tailw 2.5 --init_base $RESDIR/n6_1p4x1_msdec_sdir.json"
  "GNN --tag gnn5_n6_0p5x1_sdc_bpdec_cw  --n 6 --sched 0p5x1 --order sdc --relabel bpdec --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 800 --bs 250 --lr 0.0012 --p_cap 0.65 --tailw 2.5 --init_base $RESDIR/n6_0p5x1_sdc_bpdec.json"
  "GNN --tag gnn5_n8_2p3x1_bpdec_nocw    --n 8 --sched 2p3x1 --order csd --relabel bpdec            --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 800 --bs 250 --lr 0.0012 --p_cap 0.60 --tailw 2.5 --init_base $RESDIR/n8_2p3x1_csd_bpdec.json"
  "GNN --tag gnn5_n8_0p5x1_sdc_bpdec_cw  --n 8 --sched 0p5x1 --order sdc --relabel bpdec --clause_w --arch mp --H 128 --rounds 4 --ntrain 10000 --opt adam --steps 800 --bs 250 --lr 0.0012 --p_cap 0.60 --tailw 2.5 --init_base $RESDIR/n8_0p5x1_sdc_msdec_sdir.json"
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
    taskset -c 0-5 python3 "$P1" $args > "$LOGDIR/${tag}.log" 2>&1
  else
    taskset -c 0-5 python3 "$GNN" $args > "$LOGDIR/${tag}.log" 2>&1
  fi
  taskset -c 0-5 python3 "$AUD" "$tag" >> "$LOGDIR/${tag}.log" 2>&1 || true
  python3 /workspace/qumode_sep/oct_explore/oct23_exp/build_report.py >/dev/null 2>&1 || true
  echo "Finished $tag"
}
export -f run_job
export P1 GNN AUD LOGDIR RESDIR

printf "%s\n" "${JOBS[@]}" | xargs -I {} -P 5 bash -c 'run_job "$@"' _ {}
echo "ALL FINISH_PART1 JOBS COMPLETE"

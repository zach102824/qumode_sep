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

WAVE3_JOBS=(
  # 1. Stage-1 Gray Mixer L=4..10 (completing L=4,6,8,10,12,15 table) + N_train scaling at L=15
  "--tag p2_gray_L4         --mode shared --mixer gray --L 4  --pairs 1 --kprep 2 --ntrain8 2000  --ntrain12 500"
  "--tag p2_gray_L6         --mode shared --mixer gray --L 6  --pairs 1 --kprep 2 --ntrain8 2000  --ntrain12 500"
  "--tag p2_gray_L8         --mode shared --mixer gray --L 8  --pairs 1 --kprep 2 --ntrain8 2000  --ntrain12 500"
  "--tag p2_gray_L10        --mode shared --mixer gray --L 10 --pairs 1 --kprep 2 --ntrain8 2000  --ntrain12 500"
  "--tag p2_gray_L15_ntr500   --mode shared --mixer gray --L 15 --pairs 1 --kprep 2 --ntrain8 500   --ntrain12 200"
  "--tag p2_gray_L15_ntr5000  --mode shared --mixer gray --L 15 --pairs 1 --kprep 2 --ntrain8 5000  --ntrain12 1000"
  "--tag p2_gray_L15_ntr10000 --mode shared --mixer gray --L 15 --pairs 1 --kprep 2 --ntrain8 10000 --ntrain12 1000"

  # 2. Pure GNN on Gray Mixer + Per-Cavity 0..15 SNAP (testing whether GNN closes B90 gap to WalkSATlm tries!)
  "--tag p2_gnn_gray_L12_mp64_nocw --mode gnn --mixer gray --L 12 --pairs 1 --kprep 2 --arch mp64 --steps 600 --init_shared $RESDIR/p2_gray_L12.json"
  "--tag p2_gnn_gray_L12_mp64_cw   --mode gnn --mixer gray --L 12 --pairs 1 --kprep 2 --arch mp64 --clause_w --steps 600 --init_shared $RESDIR/p2_gray_L12.json"
  "--tag p2_gnn_gray_L15_mp64_nocw --mode gnn --mixer gray --L 15 --pairs 1 --kprep 2 --arch mp64 --steps 600 --init_shared $RESDIR/p2_gray_L15.json"
  "--tag p2_gnn_gray_L15_mp64_cw   --mode gnn --mixer gray --L 15 --pairs 1 --kprep 2 --arch mp64 --clause_w --steps 600 --init_shared $RESDIR/p2_gray_L15.json"

  # 3. Physical SNAP+D Shared BFGS N_train scaling (ntrain8 in {500, 5000, 10000} vs 2000 baseline)
  "--tag p2_shared_L12_k2_ntr500   --mode shared --L 12 --pairs 2 --kprep 2 --ntrain8 500   --ntrain12 200  --maxiter 60 --grow_from $RESDIR/p2_shared_L12_k2.json"
  "--tag p2_shared_L12_k2_ntr5000  --mode shared --L 12 --pairs 2 --kprep 2 --ntrain8 5000  --ntrain12 1000 --maxiter 60 --grow_from $RESDIR/p2_shared_L12_k2.json"
  "--tag p2_shared_L12_k2_ntr10000 --mode shared --L 12 --pairs 2 --kprep 2 --ntrain8 10000 --ntrain12 1000 --maxiter 60 --grow_from $RESDIR/p2_shared_L12_k2.json"
  "--tag p2_shared_L15_k3_ntr500   --mode shared --L 15 --pairs 3 --kprep 3 --ntrain8 500   --ntrain12 200  --maxiter 60 --grow_from $RESDIR/p2_shared_L15_k3.json"
  "--tag p2_shared_L15_k3_ntr5000  --mode shared --L 15 --pairs 3 --kprep 3 --ntrain8 5000  --ntrain12 1000 --maxiter 60 --grow_from $RESDIR/p2_shared_L15_k3.json"
  "--tag p2_shared_L15_k3_ntr10000 --mode shared --L 15 --pairs 3 --kprep 3 --ntrain8 10000 --ntrain12 1000 --maxiter 60 --grow_from $RESDIR/p2_shared_L15_k3.json"

  # 4. Physical SNAP+D Pure GNN with mp64 (H=64, r=3) at L=12 k=2 and L=15 k=3 (less overfitting than mp128)
  "--tag p2_gnn_L12_k2_mp64_nocw --mode gnn --L 12 --pairs 2 --kprep 2 --arch mp64 --steps 800 --init_shared $RESDIR/p2_shared_L12_k2.json"
  "--tag p2_gnn_L12_k2_mp64_cw   --mode gnn --L 12 --pairs 2 --kprep 2 --arch mp64 --clause_w --steps 800 --init_shared $RESDIR/p2_shared_L12_k2.json"
  "--tag p2_gnn_L15_k3_mp64_nocw --mode gnn --L 15 --pairs 3 --kprep 3 --arch mp64 --steps 800 --init_shared $RESDIR/p2_shared_L15_k3.json"
  "--tag p2_gnn_L15_k3_mp64_cw   --mode gnn --L 15 --pairs 3 --kprep 3 --arch mp64 --clause_w --steps 800 --init_shared $RESDIR/p2_shared_L15_k3.json"
)

run_one() {
  local args="$1"
  local tag=$(echo "$args" | awk '{for(i=1;i<=NF;i++) if($i=="--tag") print $(i+1)}')
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

printf "%s\n" "${WAVE3_JOBS[@]}" | xargs -I {} -P 6 bash -c 'run_one "$@"' _ {}
echo "ALL WAVE3 JOBS COMPLETE"

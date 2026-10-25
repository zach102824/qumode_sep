#!/usr/bin/env bash
# WAVE12: after WAVE11 dual ★T at n=8 (best 1.58/3.69 s3000; also 1.71/3.72 n12_3000)
# Priority: n12 B90 (need ≤6.28; best ~10.1), n16/n20 Med+B90, flatter growth — keep n8 ★T.
# - ntr10000 × n12=3000 × s2000 (longer Adam on best-growth ★T lineage)
# - ntr10000 × n12=3000 × s2400 (combine more n=12 train × longer)
# - ntr10000 × n12=2000 × s3600 (extend best ★T Adam s3000→s3600)
# - mp128 × ntr10000 × n12=2000 × s2000 (capacity × longer on Med/★T recipe)
# Keep total Part2 workers ≤4 on cores 0-5 (prior -P6 OOM). Leaves room for @qumode_checker.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_NUM_INTRA_OP_THREADS=1 JAX_NUM_INTER_OP_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
SCRIPT=/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py
RES=/workspace/qumode_sep/oct_explore/results/oct23_exp_part2
LOG=/workspace/qumode_sep/oct_explore/logs/oct23_exp_part2
INIT=$RES/p2_shared_L15_k4_grayinit.json
mkdir -p "$LOG" "$RES"

run_one() {
  local tag="$1"; shift
  if [ -f "$RES/${tag}.json" ]; then echo "Skipping $tag (done)"; return 0; fi
  if pgrep -f -- "--tag ${tag}( |$)" >/dev/null 2>&1; then echo "Skipping $tag (running)"; return 0; fi
  echo "Starting $tag ..."
  taskset -c 0-5 python3 "$SCRIPT" --tag "$tag" "$@" > "$LOG/${tag}.log" 2>&1
  python3 /workspace/qumode_sep/oct_explore/oct23_exp/build_part2_report.py >/dev/null 2>&1 || true
  echo "Finished $tag"
}

run_one p2_gnn_L15_k4_grayinit_mp64_cw_ntr10000_n12_3000_s2000 --mode gnn --L 15 --pairs 4 --kprep 3 --arch mp64 --clause_w --steps 2000 --ntrain8 10000 --ntrain12 3000 --init_shared "$INIT" &
run_one p2_gnn_L15_k4_grayinit_mp64_cw_ntr10000_n12_3000_s2400 --mode gnn --L 15 --pairs 4 --kprep 3 --arch mp64 --clause_w --steps 2400 --ntrain8 10000 --ntrain12 3000 --init_shared "$INIT" &
run_one p2_gnn_L15_k4_grayinit_mp64_cw_ntr10000_n12_2000_s3600 --mode gnn --L 15 --pairs 4 --kprep 3 --arch mp64 --clause_w --steps 3600 --ntrain8 10000 --ntrain12 2000 --init_shared "$INIT" &
run_one p2_gnn_L15_k4_grayinit_mp128_cw_ntr10000_n12_2000_s2000 --mode gnn --L 15 --pairs 4 --kprep 3 --arch mp128 --clause_w --steps 2000 --ntrain8 10000 --ntrain12 2000 --init_shared "$INIT" &
wait
echo "ALL WAVE12 JOBS COMPLETE"

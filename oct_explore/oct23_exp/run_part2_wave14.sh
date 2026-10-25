#!/usr/bin/env bash
# WAVE14: after WAVE13 complete — best phys n12 B90 8.37 (n12_3000_s2800 ★T 1.69/3.74);
#         WAVE12 still best phys ★T n8 1.63/3.47 (n12_3000_s2400) + best n12 Med 2.68;
#         WAVE11 best n8 Med 1.58 + best n16 B90 19.73; WAVE12 s2000 best n20 Med 8.82 / flattest Med 1.1517ⁿ.
#         WAVE13 s3000 ★T 1.69/3.66 trails s2800 on n12 B90; n12_4000 + mp128@s2400 no new crowns.
# Priority: push n12 B90 (need ≤6.28; best 8.37) / large-n while keeping n8 ★T.
# - ntr10000 × n12=3000 × s2600 (between best ★T s2400 and best n12 B90 s2800)
# - ntr10000 × n12=3000 × s3200 (probe past s3000 overfit)
# - ntr10000 × n12=5000 × s2800 (more n12 on winning step count; cache via get_instances)
# - mp128 × ntr10000 × n12=3000 × s2800 (capacity on best n12 B90 recipe)
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

run_one p2_gnn_L15_k4_grayinit_mp64_cw_ntr10000_n12_3000_s2600 --mode gnn --L 15 --pairs 4 --kprep 3 --arch mp64 --clause_w --steps 2600 --ntrain8 10000 --ntrain12 3000 --init_shared "$INIT" &
run_one p2_gnn_L15_k4_grayinit_mp64_cw_ntr10000_n12_3000_s3200 --mode gnn --L 15 --pairs 4 --kprep 3 --arch mp64 --clause_w --steps 3200 --ntrain8 10000 --ntrain12 3000 --init_shared "$INIT" &
run_one p2_gnn_L15_k4_grayinit_mp64_cw_ntr10000_n12_5000_s2800 --mode gnn --L 15 --pairs 4 --kprep 3 --arch mp64 --clause_w --steps 2800 --ntrain8 10000 --ntrain12 5000 --init_shared "$INIT" &
run_one p2_gnn_L15_k4_grayinit_mp128_cw_ntr10000_n12_3000_s2800 --mode gnn --L 15 --pairs 4 --kprep 3 --arch mp128 --clause_w --steps 2800 --ntrain8 10000 --ntrain12 3000 --init_shared "$INIT" &
wait
echo "ALL WAVE14 JOBS COMPLETE"

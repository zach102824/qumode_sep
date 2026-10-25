#!/usr/bin/env bash
# WAVE13: after WAVE12 complete — best phys ★T n8 1.63/3.47 (n12_3000_s2400);
#         flattest Med + best n20 Med from n12_3000_s2000 (1.65/3.59…8.82);
#         WAVE11 still best n8 Med (1.58) and best n16 B90 (19.73 from n12_3000_s1600).
#         s3600 (n12_2000) and mp128@n12_2000_s2000 missed ★T (overfit / 3.7445≰3.74).
# Priority: push n12 B90 (need ≤6.28; best 9.85) / large-n while keeping n8 ★T.
# - ntr10000 × n12=3000 × s2800 (extend best ★T Adam s2400→s2800 on winning n12_3000)
# - ntr10000 × n12=3000 × s3000 (further Adam on same lineage)
# - ntr10000 × n12=4000 × s2400 (more n=12 train on ★T recipe; cache auto-gen via get_instances)
# - mp128 × ntr10000 × n12=3000 × s2400 (capacity on winning n12_3000_s2400 recipe)
# If n12=4000 OOMs/unsupported mid-run, swap that slot to …_n12_3000_s3200 mp64.
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

run_one p2_gnn_L15_k4_grayinit_mp64_cw_ntr10000_n12_3000_s2800 --mode gnn --L 15 --pairs 4 --kprep 3 --arch mp64 --clause_w --steps 2800 --ntrain8 10000 --ntrain12 3000 --init_shared "$INIT" &
run_one p2_gnn_L15_k4_grayinit_mp64_cw_ntr10000_n12_3000_s3000 --mode gnn --L 15 --pairs 4 --kprep 3 --arch mp64 --clause_w --steps 3000 --ntrain8 10000 --ntrain12 3000 --init_shared "$INIT" &
run_one p2_gnn_L15_k4_grayinit_mp64_cw_ntr10000_n12_4000_s2400 --mode gnn --L 15 --pairs 4 --kprep 3 --arch mp64 --clause_w --steps 2400 --ntrain8 10000 --ntrain12 4000 --init_shared "$INIT" &
run_one p2_gnn_L15_k4_grayinit_mp128_cw_ntr10000_n12_3000_s2400 --mode gnn --L 15 --pairs 4 --kprep 3 --arch mp128 --clause_w --steps 2400 --ntrain8 10000 --ntrain12 3000 --init_shared "$INIT" &
wait
echo "ALL WAVE13 JOBS COMPLETE"

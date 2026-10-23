#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_NUM_INTRA_OP_THREADS=1
export JAX_NUM_INTER_OP_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"

SCRIPT="/workspace/qumode_sep/oct_explore/oct23_exp/part1_2cav.py"
LOGDIR="/workspace/qumode_sep/oct_explore/logs/oct23_exp"
RESDIR="/workspace/qumode_sep/oct_explore/results/oct23_exp"
mkdir -p "$LOGDIR" "$RESDIR"

COMMANDS=(
  # 1. N_train exploration at n=6 (1p4x1, csd, sd, relabel=none and relabel=bpdec)
  "--tag n6_ntrain500_none   --n 6 --sched 1p4x1 --order csd --gate sd --ntrain 500   --nscreen 500  --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"
  "--tag n6_ntrain2000_none  --n 6 --sched 1p4x1 --order csd --gate sd --ntrain 2000  --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"
  "--tag n6_ntrain5000_none  --n 6 --sched 1p4x1 --order csd --gate sd --ntrain 5000  --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"
  "--tag n6_ntrain10000_none --n 6 --sched 1p4x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"

  "--tag n6_ntrain500_bpdec   --n 6 --sched 1p4x1 --order csd --gate sd --ntrain 500   --nscreen 500  --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"
  "--tag n6_ntrain2000_bpdec  --n 6 --sched 1p4x1 --order csd --gate sd --ntrain 2000  --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"
  "--tag n6_ntrain5000_bpdec  --n 6 --sched 1p4x1 --order csd --gate sd --ntrain 5000  --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"
  "--tag n6_ntrain10000_bpdec --n 6 --sched 1p4x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"

  # 2. Schedule comparison at n=6 (ntrain=10000, both relabel=none and relabel=bpdec, csd and sdc)
  "--tag n6_0p5x1_csd_none  --n 6 --sched 0p5x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"
  "--tag n6_0p5x1_sdc_none  --n 6 --sched 0p5x1 --order sdc --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"
  "--tag n6_2p3x1_csd_none  --n 6 --sched 2p3x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"
  "--tag n6_1p2x2_csd_none  --n 6 --sched 1p2x2 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"

  "--tag n6_0p5x1_csd_bpdec --n 6 --sched 0p5x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"
  "--tag n6_0p5x1_sdc_bpdec --n 6 --sched 0p5x1 --order sdc --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"
  "--tag n6_2p3x1_csd_bpdec --n 6 --sched 2p3x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"
  "--tag n6_1p2x2_csd_bpdec --n 6 --sched 1p2x2 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"

  # 3. Relabeling ablation at n=6 (1p4x1, csd, sd, ntrain=10000: bias, biaspol, bppol, bpdec, rr4)
  "--tag n6_1p4x1_bias    --n 6 --sched 1p4x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bias --asym"
  "--tag n6_1p4x1_biaspol --n 6 --sched 1p4x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel biaspol --asym"
  "--tag n6_1p4x1_bppol   --n 6 --sched 1p4x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bppol --asym"

  # 4. Gate comparison at n=6 (ECD ancilla vs ECDT hybrid transmon+cavity, 0p5x1 and 1p4x1)
  "--tag n6_0p5x1_ecd_none   --n 6 --sched 0p5x1 --order csd --gate ecd  --ntrain 10000 --nscreen 1000 --nstarts 12 --screen_it 80 --refine_top 2 --maxiter 300 --relabel none"
  "--tag n6_1p4x1_ecd_none   --n 6 --sched 1p4x1 --order csd --gate ecd  --ntrain 10000 --nscreen 1000 --nstarts 12 --screen_it 80 --refine_top 2 --maxiter 300 --relabel none"
  "--tag n6_1p4x1_ecdt_none  --n 6 --sched 1p4x1 --order csd --gate ecdt --ntrain 10000 --nscreen 1000 --nstarts 12 --screen_it 80 --refine_top 2 --maxiter 300 --relabel none"
  "--tag n6_1p4x1_ecd_bpdec  --n 6 --sched 1p4x1 --order csd --gate ecd  --ntrain 10000 --nscreen 1000 --nstarts 12 --screen_it 80 --refine_top 2 --maxiter 300 --relabel bpdec --asym"
  "--tag n6_1p4x1_ecdt_bpdec --n 6 --sched 1p4x1 --order csd --gate ecdt --ntrain 10000 --nscreen 1000 --nstarts 12 --screen_it 80 --refine_top 2 --maxiter 300 --relabel bpdec --asym"

  # 5. N_train exploration at n=8 (1p4x1, csd, sd, relabel=none and relabel=bpdec)
  "--tag n8_ntrain500_none   --n 8 --sched 1p4x1 --order csd --gate sd --ntrain 500   --nscreen 500  --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"
  "--tag n8_ntrain2000_none  --n 8 --sched 1p4x1 --order csd --gate sd --ntrain 2000  --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"
  "--tag n8_ntrain5000_none  --n 8 --sched 1p4x1 --order csd --gate sd --ntrain 5000  --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"
  "--tag n8_ntrain10000_none --n 8 --sched 1p4x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"

  "--tag n8_ntrain500_bpdec   --n 8 --sched 1p4x1 --order csd --gate sd --ntrain 500   --nscreen 500  --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"
  "--tag n8_ntrain2000_bpdec  --n 8 --sched 1p4x1 --order csd --gate sd --ntrain 2000  --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"
  "--tag n8_ntrain5000_bpdec  --n 8 --sched 1p4x1 --order csd --gate sd --ntrain 5000  --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"
  "--tag n8_ntrain10000_bpdec --n 8 --sched 1p4x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"

  # 6. Schedule & Relabel comparison at n=8 (ntrain=10000)
  "--tag n8_0p5x1_csd_none  --n 8 --sched 0p5x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"
  "--tag n8_0p5x1_sdc_none  --n 8 --sched 0p5x1 --order sdc --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"
  "--tag n8_2p3x1_csd_none  --n 8 --sched 2p3x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"
  "--tag n8_1p2x2_csd_none  --n 8 --sched 1p2x2 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel none"

  "--tag n8_0p5x1_csd_bpdec --n 8 --sched 0p5x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"
  "--tag n8_0p5x1_sdc_bpdec --n 8 --sched 0p5x1 --order sdc --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"
  "--tag n8_2p3x1_csd_bpdec --n 8 --sched 2p3x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"
  "--tag n8_1p2x2_csd_bpdec --n 8 --sched 1p2x2 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bpdec --asym"

  "--tag n8_1p4x1_bias    --n 8 --sched 1p4x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bias --asym"
  "--tag n8_1p4x1_biaspol --n 8 --sched 1p4x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel biaspol --asym"
  "--tag n8_1p4x1_bppol   --n 8 --sched 1p4x1 --order csd --gate sd --ntrain 10000 --nscreen 1000 --nstarts 16 --screen_it 100 --refine_top 2 --maxiter 200 --relabel bppol --asym"

  # 7. Gate comparison at n=8 (ECD ancilla vs ECDT hybrid transmon+cavity)
  "--tag n8_0p5x1_ecd_none   --n 8 --sched 0p5x1 --order csd --gate ecd  --ntrain 10000 --nscreen 1000 --nstarts 12 --screen_it 80 --refine_top 2 --maxiter 300 --relabel none"
  "--tag n8_1p4x1_ecd_none   --n 8 --sched 1p4x1 --order csd --gate ecd  --ntrain 10000 --nscreen 1000 --nstarts 12 --screen_it 80 --refine_top 2 --maxiter 300 --relabel none"
  "--tag n8_1p4x1_ecdt_none  --n 8 --sched 1p4x1 --order csd --gate ecdt --ntrain 10000 --nscreen 1000 --nstarts 12 --screen_it 80 --refine_top 2 --maxiter 300 --relabel none"
  "--tag n8_1p4x1_ecd_bpdec  --n 8 --sched 1p4x1 --order csd --gate ecd  --ntrain 10000 --nscreen 1000 --nstarts 12 --screen_it 80 --refine_top 2 --maxiter 300 --relabel bpdec --asym"
  "--tag n8_1p4x1_ecdt_bpdec --n 8 --sched 1p4x1 --order csd --gate ecdt --ntrain 10000 --nscreen 1000 --nstarts 12 --screen_it 80 --refine_top 2 --maxiter 300 --relabel bpdec --asym"
)

run_one() {
  local args="$1"
  local tag=$(echo "$args" | awk '{for(i=1;i<=NF;i++) if($i=="--tag") print $(i+1)}')
  if [ -f "$RESDIR/${tag}.json" ]; then
    echo "Skipping $tag (already done)"
    return 0
  fi
  echo "Starting $tag ..."
  # Pin to cores 0-4 (core 5 for companion GNN; 6-7 free for @qumode_checker)
  taskset -c 0-4 python3 "$SCRIPT" $args > "$LOGDIR/${tag}.log" 2>&1
  echo "Finished $tag"
}
export -f run_one
export SCRIPT LOGDIR RESDIR

printf "%s\n" "${COMMANDS[@]}" | xargs -I {} -P 5 bash -c 'run_one "$@"' _ {}
echo "ALL WAVE 1 SHARED JOBS COMPLETE"

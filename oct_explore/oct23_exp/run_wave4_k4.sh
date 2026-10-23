#!/usr/bin/env bash
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_NUM_INTRA_OP_THREADS=1 JAX_NUM_INTER_OP_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
SCRIPT=/workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py
COMP=/workspace/qumode_sep/oct_explore/oct23_exp/compile_gray_to_sd.py
RES=/workspace/qumode_sep/oct_explore/results/oct23_exp_part2
LOG=/workspace/qumode_sep/oct_explore/logs/oct23_exp_part2

taskset -c 0-5 python3 $COMP --pairs 4 --kprep 3 --starts 20 --maxiter_layer 400 --maxiter_traj 120 --out $RES/init_gray_compiled_L15_k4.json > $LOG/compile_gray_L15_k4.log 2>&1
taskset -c 0-5 python3 $SCRIPT --tag p2_shared_L15_k4_grayinit --mode shared --L 15 --pairs 4 --kprep 3 --maxiter 200 --grow_from $RES/init_gray_compiled_L15_k4.json > $LOG/p2_shared_L15_k4_grayinit.log 2>&1
taskset -c 0-5 python3 $SCRIPT --tag p2_gnn_L15_k4_grayinit_mp64_cw --mode gnn --L 15 --pairs 4 --kprep 3 --arch mp64 --clause_w --steps 800 --init_shared $RES/p2_shared_L15_k4_grayinit.json > $LOG/p2_gnn_L15_k4_grayinit_mp64_cw.log 2>&1

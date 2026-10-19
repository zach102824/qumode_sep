#!/bin/bash
# Oct 19 (Zach 09:3x CST): S_g4_snapF_s1 with vacuum start (--psi0 vac), warm from its 3 trained inits, 3 parallel jobs x 3/3/2 threads on cores 0-7 (cap raised to 8 at 09:41 CST),
# then reselect_arch (n=10/14/20 fresh val, 600/300/100, base 37e9) over the 3 inits and TEST winner on TEB n=10/14/20.
cd /workspace/qumode_sep/oct_explore
PY=/usr/bin/python3.13
ENV="XLA_PYTHON_CLIENT_PREALLOCATE=false"
pids=()
TH=(3 3 2)
for k in 0 1 2; do
  t=${TH[$k]}
  taskset -c 0-7 env $ENV OMP_NUM_THREADS=$t XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$t" \
    $PY v3/gnn_snap.py --tag S_g4_snapF_s1_vac_w$k --snap_form free --L 4 --clause_w --var_mix --snap std \
    --warm_same results/v3/S_g4_snapF_s1.json --warm_ids $k --ninit 1 --maxiter 1000 --feat formula --chunk 125 --seed 1 --psi0 vac \
    > logs/v3/S_g4_snapF_s1_vac_w$k.log 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
taskset -c 0-7 env $ENV OMP_NUM_THREADS=8 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=8" \
  $PY v3/reselect_arch.py --tag S_g4_snapF_s1_vac --src S_g4_snapF_s1_vac_w0 S_g4_snapF_s1_vac_w1 S_g4_snapF_s1_vac_w2 --test 10 14 20 \
  > logs/v3/S_g4_snapF_s1_vac.log 2>&1
echo VACDONE >> logs/v3/S_g4_snapF_s1_vac.log

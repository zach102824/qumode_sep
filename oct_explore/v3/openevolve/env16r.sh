# common env for net16r (sourced by run/driver/test scripts)
cd /workspace/qumode_sep/oct_explore/v3/openevolve
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OE_TRAIN_N=8,12 OE_VAL_N=16 OE_CHECK_N=16 OE_NVAL=60 OE_BUDGET=180 OE_MEM_MB=3000 OE_S1_BUDGET=60 OE_S1_MARGIN=${OE_S1_MARGIN:-0.35}
export GOV_SLOTS=2 GOV_FLOOR_MB=3200 GOV_NEED_MB=2600

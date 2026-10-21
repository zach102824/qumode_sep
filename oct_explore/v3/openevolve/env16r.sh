# common env for net16r (sourced by run/driver/test scripts)
cd /workspace/qumode_sep/oct_explore/v3/openevolve
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OE_TRAIN_N=8,12 OE_VAL_N=16 OE_CHECK_N=16 OE_NVAL=60 OE_BUDGET=180 OE_MEM_MB=3000 OE_S1_BUDGET=60 OE_S1_MARGIN=${OE_S1_MARGIN:-0.35}
export GOV_SLOTS=2 GOV_FLOOR_MB=3200 GOV_NEED_MB=2600
# net16r is TOP PRIORITY: nice -5 + oom_score_adj -500 for every net16r process (children inherit; watchdog re-applies)
protect16r(){ for p in $(pgrep -f "net16r|evaluator_net16r|eval_governor16r|cursor_adapter.py|openevolve.cli runs/net16r|chunked_check.py"); do
  sudo -n renice -n -5 -p $p >/dev/null 2>&1; sudo -n choom -n -500 -p $p >/dev/null 2>&1 || echo -500 > /proc/$p/oom_score_adj 2>/dev/null; done; }

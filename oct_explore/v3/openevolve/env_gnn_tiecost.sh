# common env for gnn_tiecost
cd /workspace/qumode_sep/oct_explore/v3/openevolve
export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OE_L=${OE_L:-5}
export OE_SCREEN_STEPS=${OE_SCREEN_STEPS:-200}
export OE_FULL_STEPS_CAP=${OE_FULL_STEPS_CAP:-1000}
export OE_NTRAIN_SCREEN=${OE_NTRAIN_SCREEN:-2000}
export OE_MEM_MB=${OE_MEM_MB:-4500}
export GOV_SLOTS=${GOV_SLOTS:-1} GOV_FLOOR_MB=0 GOV_NEED_MB=0
protect_tiecost(){ for p in $(pgrep -f "gnn_tiecost|evaluator_gnn_tiecost|cursor_adapter_tiecost|openevolve.cli runs/gnn_tiecost"); do
  sudo -n renice -n -5 -p $p >/dev/null 2>&1; sudo -n choom -n -500 -p $p >/dev/null 2>&1 || echo -500 > /proc/$p/oom_score_adj 2>/dev/null; done; }

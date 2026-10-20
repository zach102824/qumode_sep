#!/bin/bash
# OpenEvolve on the network (2 cores 5-6). usage: ITER=10 OUT=runs/net_smoke ./run_net.sh
cd "$(dirname "$0")"; export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
: "${CURSOR_API_KEY:?CURSOR_API_KEY must be set}"
export OE_BUDGET=${OE_BUDGET:-300}
curl -s -m 2 127.0.0.1:8765/v1/models >/dev/null || { setsid nohup python3.13 cursor_adapter.py --port 8765 --model composer-2.5 --conc 2 > logs/adapter.log 2>&1 & sleep 3; }
exec taskset -c 5-6 python3.13 -m openevolve.cli net/initial_net.py net/evaluator_net.py --config config_net.yaml --iterations ${ITER:-10} --output ${OUT:-runs/net_smoke} ${EXTRA}

#!/bin/bash
# smoke: adapter + few OpenEvolve iterations. Never prints the key.
cd "$(dirname "$0")"; export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
: "${CURSOR_API_KEY:?CURSOR_API_KEY must be set}"
curl -s -m 2 127.0.0.1:8765/v1/models >/dev/null || { setsid nohup python3.13 cursor_adapter.py --port 8765 --model composer-2.5 --conc 2 > logs/adapter.log 2>&1 & sleep 3; }
ITER=${ITER:-5}
exec taskset -c 0-6 python3.13 -m openevolve.cli initial_program.py evaluator.py --config config.yaml --iterations $ITER --output runs/smoke

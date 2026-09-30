#!/usr/bin/env bash
# SPSA step scan at n=16 (F1): ry0 (exact Gibbs) + hea1 at 400/600/800 steps; resumable.
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD OMP_NUM_THREADS=1
PY=${PY:-/workspace/venv-qumode/bin/python}
for S in 400 600 800; do
  $PY noiseless/run_scaling.py --tag steps$S --steps $S --arms ry0,hea1 --n 16 --workers 8
done
echo SCAN_DONE

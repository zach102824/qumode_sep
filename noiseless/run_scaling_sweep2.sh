#!/usr/bin/env bash
# Phase 2b: RY-only with exact (hi/lo factorized) Gibbs cost at n = 24, 28, plus
# validation of that path against the per-state exact cost at n = 16, 20. Resumable.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1
PY=${PY:-/workspace/venv-qumode/bin/python}
TAG=${TAG:-main}
W=${W:-8}
$PY noiseless/run_scaling.py --tag $TAG --workers $W --arms ry0_split --n 16,20,24
$PY noiseless/run_scaling.py --tag $TAG --workers $W --arms ry0_split --n 28
$PY noiseless/run_scaling.py --tag $TAG --summarize > /dev/null
echo SWEEP2_DONE

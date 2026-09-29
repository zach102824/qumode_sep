#!/usr/bin/env bash
# Full scaling sweep (Phase 2). Resumable: re-running skips checkpointed jobs.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1
PY=${PY:-/workspace/venv-qumode/bin/python}
TAG=${TAG:-main}
W=${W:-8}
$PY noiseless/run_scaling.py --tag $TAG --workers $W --arms ry0,classical --n 8,12,16,20,24,28
$PY noiseless/run_scaling.py --tag $TAG --workers $W --arms ry0_sampled --n 16,20
$PY noiseless/run_scaling.py --tag $TAG --workers $W --arms hea1 --n 8,12,16,20
$PY noiseless/run_scaling.py --tag $TAG --workers $W --arms hea2 --n 8,12,16,20
$PY noiseless/run_scaling.py --tag $TAG --summarize > /dev/null
echo SWEEP_DONE

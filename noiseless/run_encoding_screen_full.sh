#!/usr/bin/env bash
# Full brute-force encoding screen: all 20160 assignment classes x 3 paired inits x 20 four_sat
# Hamiltonians (n=8, jp, tuned growth + SPSA-Adam), 6 workers (default; WORKERS env overrides). Hamiltonians run sequentially,
# each a resumable run_encoding_screen.py invocation (JSONL checkpoint, finished trials skipped),
# so re-running this script resumes. Up to 3 passes, so a crashed Hamiltonian is retried.
#
# Launch detached (survives the session):
#   setsid nohup noiseless/run_encoding_screen_full.sh > logs/encoding_screen_full.log 2>&1 < /dev/null &
# Status: python noiseless/encoding_screen_status.py
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
PY=${PY:-/workspace/venv-qumode/bin/python}
TAG=${TAG:-full}
WORKERS=${WORKERS:-6}
INITS=${INITS:-3}
NH=${NH:-20}
export PYTHONPATH="$REPO" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs "noiseless/results/encoding_screen/$TAG"
PIDFILE="noiseless/results/encoding_screen/$TAG/driver.pid"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "driver already running (pid $(cat "$PIDFILE"))"; exit 1
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
echo "[$(date '+%F %T %Z')] driver pid $$ start tag=$TAG workers=$WORKERS inits=$INITS NH=$NH commit $(git rev-parse --short HEAD)"
for pass in 1 2 3; do
  fail=0
  for ((h = 0; h < NH; h++)); do
    echo "[$(date '+%F %T %Z')] pass $pass H $h start"
    if "$PY" noiseless/run_encoding_screen.py --ham "$h" --assignments all --inits "$INITS" \
        --workers "$WORKERS" --tag "$TAG"; then
      echo "[$(date '+%F %T %Z')] pass $pass H $h done"
    else
      echo "[$(date '+%F %T %Z')] pass $pass H $h FAILED (rc=$?)"; fail=1
    fi
  done
  [[ $fail == 0 ]] && break
done
echo "[$(date '+%F %T %Z')] driver finished (last pass fail=$fail)"

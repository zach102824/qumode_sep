#!/bin/bash
# Rebuild RELAYOUT_SCALING_EXPLORE.md = head (hand-written findings) + generated table + tail (status / next steps)
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=/workspace/qumode_sep
{ cat noiseless/results/RELAYOUT_SCALING_EXPLORE_head.md
  echo; echo "## All full runs (100 trials each, sorted by evals)"
  python noiseless/scaling_explore_table.py
  echo; cat noiseless/results/RELAYOUT_SCALING_EXPLORE_tail.md; } > noiseless/results/RELAYOUT_SCALING_EXPLORE.md

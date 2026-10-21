#!/bin/bash
source /workspace/qumode_sep/oct_explore/v3/openevolve/env16r.sh
export OE_RUN_DIR=$PWD/runs/net16r_test; rm -f $OE_RUN_DIR/s1_state.json
t=$(date +%s); python3.13 net/evaluator_net16r.py net/initial_net16r.py > $OE_RUN_DIR/base.json 2> $OE_RUN_DIR/base.err; echo "base done $(( $(date +%s)-t ))s"
t=$(date +%s); python3.13 net/evaluator_net16r.py runs/net16r_test/bad_candidate.py > $OE_RUN_DIR/bad.json 2> $OE_RUN_DIR/bad.err; echo "bad done $(( $(date +%s)-t ))s"

#!/bin/bash
# stop queue1 right after tunedsa10 n18 (skip its satlike10 n20, a variant that already failed n18)
cd /workspace/qumode_sep
until grep -q "^| tunedsa10 | 18 |" noiseless/results/OVERNIGHT_IMPROVEMENTS.md; do sleep 20; done
sleep 5; pkill -f "logs/ovn/queue1.sh"; sleep 2; pkill -f "run_variant.sh satlike10 10 20"; pkill -f "outdir noiseless/results/relayout_runs/ovn_satlike10 .*--n 20"; touch logs/ovn/queue1.done

#!/usr/bin/env bash
set -euo pipefail
RES=/workspace/qumode_sep/oct_explore/results/oct23_exp_part2
LOG=/workspace/qumode_sep/oct_explore/logs/oct23_exp_part2
TAG=p2_gnn_L15_k4_grayinit_mp64_cw_ntr10000_n12_2000_s2000
mkdir -p "$LOG"
echo "$(date '+%H:%M:%S') chain: waiting for WAVE10 $TAG.json and runner exit..." >> "$LOG/chain_wave10_to_11.log"
# Wait until result JSON exists
while [ ! -f "$RES/${TAG}.json" ]; do sleep 30; done
echo "$(date '+%H:%M:%S') chain: $TAG.json present; waiting for wave10 runner to exit..." >> "$LOG/chain_wave10_to_11.log"
# Wait for wave10 bash to finish (avoid overlapping workers)
while pgrep -f 'run_part2_wave10.sh' >/dev/null 2>&1; do sleep 15; done
# Brief settle for report rebuild from wave10
sleep 5
python3 /workspace/qumode_sep/oct_explore/oct23_exp/build_part2_report.py >> "$LOG/chain_wave10_to_11.log" 2>&1 || true
echo "$(date '+%H:%M:%S') chain: starting WAVE11..." >> "$LOG/chain_wave10_to_11.log"
cd /workspace/qumode_sep
nohup bash oct_explore/oct23_exp/run_part2_wave11.sh >> "$LOG/wave11_runner.log" 2>&1 &
echo "$(date '+%H:%M:%S') chain: WAVE11 launched pid $!" >> "$LOG/chain_wave10_to_11.log"

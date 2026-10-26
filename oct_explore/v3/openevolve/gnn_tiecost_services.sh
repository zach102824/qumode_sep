#!/bin/bash
source /workspace/qumode_sep/oct_explore/v3/openevolve/env_gnn_tiecost.sh
OUT=runs/gnn_tiecost
mkdir -p logs "$OUT"
# ensure CURSOR_API_KEY present (from env or box-secrets) without printing it
if [ -z "$CURSOR_API_KEY" ]; then
  export CURSOR_API_KEY=$(python3.13 -c 'import json;print(json.load(open("/home/box/sand-data/box-secrets.json"))["card"]["CURSOR_API_KEY"])')
fi
curl -s -m 3 127.0.0.1:8767/v1/models >/dev/null || {
  setsid nohup python3.13 cursor_adapter_tiecost.py --port 8767 --model gnn_tiecost \
    --code_model cursor-grok-4.6-high-fast --idea_model claude-opus-5-5-high \
    --idea_every 5 --run_dir $OUT --conc 2 \
    >> logs/adapter_tiecost.log 2>&1 < /dev/null &
  sleep 4
  echo "$(date '+%F %T') started adapter :8767" >> $OUT/driver.log
}

#!/bin/bash
# idempotent: (re)start adapter (net16r mode), RAM governor, checkpoint checker if not running
source /workspace/qumode_sep/oct_explore/v3/openevolve/env16r.sh; OUT=runs/net16r
curl -s -m 3 127.0.0.1:8765/v1/models >/dev/null || { setsid nohup python3.13 cursor_adapter.py --port 8765 --model net16r --code_model cursor-grok-4.6-high-fast --idea_model claude-opus-5-5-medium --idea_every 5 --run_dir $OUT --conc 3 >> logs/adapter16r.log 2>&1 < /dev/null & sleep 4; echo "$(date '+%F %T') started adapter" >> $OUT/driver.log; }
pgrep -f eval_governor16r.py >/dev/null || { setsid nohup python3.13 eval_governor16r.py >> logs/eval_governor16r.log 2>&1 < /dev/null & echo "$(date '+%F %T') started governor" >> $OUT/driver.log; }
pgrep -f net16r_checker.sh >/dev/null || { setsid nohup bash net16r_checker.sh >> $OUT/checker.log 2>&1 < /dev/null & echo "$(date '+%F %T') started checker" >> $OUT/driver.log; }

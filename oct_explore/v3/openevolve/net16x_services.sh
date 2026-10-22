#!/bin/bash
# idempotent: (re)start adapter (net16x mode), RAM governor, checkpoint checker if not running
source /workspace/qumode_sep/oct_explore/v3/openevolve/env16x.sh; OUT=runs/net16x
curl -s -m 3 127.0.0.1:8766/v1/models >/dev/null || { setsid nohup python3.13 cursor_adapter16x.py --port 8766 --model net16x --code_model cursor-grok-4.6-high-fast --idea_model claude-opus-5-5-medium --idea_every 5 --run_dir $OUT --conc 3 >> logs/adapter16x.log 2>&1 < /dev/null & sleep 4; echo "$(date '+%F %T') started adapter" >> $OUT/driver.log; }
pgrep -f eval_governor16x.py >/dev/null || { setsid nohup python3.13 eval_governor16x.py >> logs/eval_governor16x.log 2>&1 < /dev/null & echo "$(date '+%F %T') started governor" >> $OUT/driver.log; }
pgrep -f net16x_checker.sh >/dev/null || { setsid nohup bash net16x_checker.sh >> $OUT/checker.log 2>&1 < /dev/null & echo "$(date '+%F %T') started checker" >> $OUT/driver.log; }

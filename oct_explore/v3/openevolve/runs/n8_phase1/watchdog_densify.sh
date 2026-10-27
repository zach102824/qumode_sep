#!/bin/bash
RUN=/workspace/qumode_sep/oct_explore/v3/openevolve/runs/n8_phase1
LOG=$RUN/watchdog_densify.log
cd "$RUN"
while true; do
  if [ -f "$RUN/DENSE_DONE" ]; then
    echo "$(date '+%F %T') DENSE_DONE — exit" >> "$LOG"
    exit 0
  fi
  # Prefer existing process by name
  LIVE=$(pgrep -f 'python3 -u densify_L5.py' | head -1 || true)
  if [ -n "$LIVE" ]; then
    echo "$LIVE" > "$RUN/densify.pid"
  else
    echo "$(date '+%F %T') densify dead — restarting" >> "$LOG"
    nohup python3 -u densify_L5.py >> densify_L5.log 2>&1 &
    echo $! > "$RUN/densify.pid"
    echo "$(date '+%F %T') started PID $(cat $RUN/densify.pid)" >> "$LOG"
    sleep 90  # avoid double-start race
  fi
  sleep 60
done

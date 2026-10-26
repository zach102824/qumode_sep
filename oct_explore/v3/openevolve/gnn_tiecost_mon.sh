#!/bin/bash
# quick status for gnn_tiecost
cd /workspace/qumode_sep/oct_explore/v3/openevolve
OUT=runs/gnn_tiecost
echo "=== $(date '+%F %T %z') gnn_tiecost ==="
echo "MemAvailable=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB"
for name in driver cli adapter watchdog; do
  case $name in
    driver) pat="gnn_tiecost_driver";;
    cli) pat="openevolve.cli runs/gnn_tiecost";;
    adapter) pat="cursor_adapter_tiecost";;
    watchdog) pat="gnn_tiecost_watchdog";;
  esac
  if pgrep -f "$pat" >/dev/null; then echo "$name ALIVE pids=$(pgrep -d, -f "$pat")"; else echo "$name DEAD"; fi
done
[ -f $OUT/STATUS.md ] && cat $OUT/STATUS.md
[ -f $OUT/driver.log ] && echo "--- driver.log (tail) ---" && tail -8 $OUT/driver.log
[ -f $OUT/evolve.log ] && echo "--- evolve.log (tail) ---" && tail -12 $OUT/evolve.log
[ -f logs/adapter_tiecost.log ] && echo "--- adapter (tail) ---" && tail -8 logs/adapter_tiecost.log

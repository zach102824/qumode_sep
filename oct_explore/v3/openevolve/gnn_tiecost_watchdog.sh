#!/bin/bash
source /workspace/qumode_sep/oct_explore/v3/openevolve/env_gnn_tiecost.sh
OUT=runs/gnn_tiecost; echo $$ > $OUT/watchdog.pid
while [ ! -f $OUT/gnn_tiecost.STOP ]; do
  if ! { [ -f $OUT/driver.pid ] && kill -0 $(cat $OUT/driver.pid) 2>/dev/null; }; then
    setsid nohup bash gnn_tiecost_driver.sh > /dev/null 2>&1 < /dev/null &
    echo "$(date '+%F %T') watchdog: (re)started driver" >> $OUT/driver.log
  fi
  bash gnn_tiecost_services.sh
  echo "$(date '+%F %T') watchdog ok avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB" > $OUT/watchdog.last
  protect_tiecost
  sleep 300
done

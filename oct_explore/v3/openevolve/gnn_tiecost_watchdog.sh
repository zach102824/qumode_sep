#!/bin/bash
source /workspace/qumode_sep/oct_explore/v3/openevolve/env_gnn_tiecost.sh
OUT=runs/gnn_tiecost; mkdir -p $OUT; echo $$ > $OUT/watchdog.pid
while [ ! -f $OUT/gnn_tiecost.STOP ]; do
  drv_alive=0; cli_alive=0
  [ -f $OUT/driver.pid ] && kill -0 $(cat $OUT/driver.pid) 2>/dev/null && drv_alive=1
  pgrep -f 'openevolve.cli runs/gnn_tiecost' >/dev/null && cli_alive=1
  if [ $drv_alive -eq 0 ]; then
    # avoid double-start race
    sleep 2
    pgrep -f 'gnn_tiecost_driver.sh' >/dev/null || {
      setsid nohup bash gnn_tiecost_driver.sh > /dev/null 2>&1 < /dev/null &
      echo "$(date '+%F %T') watchdog: (re)started driver" >> $OUT/driver.log
    }
  fi
  bash gnn_tiecost_services.sh
  echo "$(date '+%F %T') watchdog ok drv=$drv_alive cli=$cli_alive avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB" > $OUT/watchdog.last
  protect_tiecost
  sleep 300
done

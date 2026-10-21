#!/bin/bash
# cron-free watchdog (run under setsid): every 5 min make sure driver + services are alive; exits when net16r.STOP exists.
source /workspace/qumode_sep/oct_explore/v3/openevolve/env16r.sh; OUT=runs/net16r; echo $$ > $OUT/watchdog.pid
while [ ! -f $OUT/net16r.STOP ]; do
  if ! { [ -f $OUT/driver.pid ] && kill -0 $(cat $OUT/driver.pid) 2>/dev/null; }; then
    setsid nohup bash net16r_driver.sh > /dev/null 2>&1 < /dev/null & echo "$(date '+%F %T') watchdog: (re)started driver" >> $OUT/driver.log; fi
  bash net16r_services.sh
  echo "$(date '+%F %T') watchdog ok avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB" > $OUT/watchdog.last
  protect16r; sleep 300
done

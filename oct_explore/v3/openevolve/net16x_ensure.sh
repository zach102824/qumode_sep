#!/bin/bash
# idempotent reboot hook: start net16x watchdog if not running and no STOP file
cd /workspace/qumode_sep/oct_explore/v3/openevolve || exit 0
[ -f runs/net16x/net16x.STOP ] && exit 0
pgrep -f net16x_watchdog.sh >/dev/null && exit 0
setsid nohup bash net16x_watchdog.sh >/dev/null 2>&1 < /dev/null &
echo "$(date '+%F %T') ensure: started watchdog" >> runs/net16x/driver.log

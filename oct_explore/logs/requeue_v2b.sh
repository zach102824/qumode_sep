#!/bin/bash
# wait for NN_v2b L4 tests (incl n20) to be saved, then kill so watchdog relaunches with diverse-restart code for L8
cd /workspace/qumode_sep/oct_explore
for i in $(seq 1 720); do
  if grep -q "test 8+10+12+14|L4|n20" logs/NN_v2b.log; then
    sleep 5; P=$(pgrep -f -- "--tag NN_v2b( |$)")
    [[ -n "$P" ]] && kill $P && echo "$(date '+%F %T') requeue_v2b: killed $P after L4 (old code had identical restarts)" >> logs/watchdog_nn.log
    exit 0
  fi
  sleep 30
done

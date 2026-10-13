#!/bin/bash
cd /workspace/qumode_sep
# start after queue2 finishes (keeps 8 cores for current jobs)
while [ ! -f logs/ovn/queue2.done ]; do [ -f logs/ovn/STOP ] && exit 0; sleep 60; done
# queue3 may already be running; wait for it too so we don't oversubscribe
while [ ! -f logs/ovn/queue3.done ]; do [ -f logs/ovn/STOP ] && exit 0; sleep 60; done
exec logs/ovn/queue4.sh

#!/bin/bash
source /workspace/qumode_sep/oct_explore/v3/openevolve/env16r.sh
while [ ! -f runs/net16r/net16r.STOP ]; do taskset -c 6-7 python3.13 net16r_checker.py; sleep 120; done

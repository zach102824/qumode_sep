#!/bin/bash
source /workspace/qumode_sep/oct_explore/v3/openevolve/env16x.sh
while [ ! -f runs/net16x/net16x.STOP ]; do taskset -c 6-7 python3.13 net16x_checker.py; sleep 120; done

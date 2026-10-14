#!/bin/bash
cd /workspace/qumode_sep; export W=8
logs/shotfix/run.sh V2 100 14 16
logs/shotfix/run.sh V1 1000 14 16
logs/shotfix/run.sh V0 100 14 16
echo ALLDONE

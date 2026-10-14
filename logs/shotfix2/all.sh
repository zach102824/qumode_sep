#!/bin/bash
cd /workspace/qumode_sep; export W=8
for V in R1 R2 R3; do logs/shotfix2/run.sh $V 10 12 14 16; done
logs/shotfix2/run.sh B100 12
echo ALLDONE

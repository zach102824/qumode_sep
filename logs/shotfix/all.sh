#!/bin/bash
cd /workspace/qumode_sep
for v in V0 V1 V2 V3; do logs/shotfix/run.sh $v 100 8 10; done
for v in V1 V3; do logs/shotfix/run.sh $v 1000 8 10; done
echo ALLDONE

#!/bin/bash
cd /workspace/qumode_sep; export W=8
B="--shot-readout lowE --shot-crn"
EXTRA="--shots 1000 --shot-s0 50 --shot-eta-anneal 16 $B" logs/shotfix2/run.sh C23 10 12 14 16
EXTRA="--shots 1000 --shot-s0 50 --shot-eta-anneal 16 --xbeta 1 $B" logs/shotfix2/run.sh C123 10 12 14 16
EXTRA="--shots 1000 --shot-eta-anneal 16 $B" logs/shotfix2/run.sh C3S1000 10 12 14 16
EXTRA="--shots 3000 --shot-s0 100 --shot-eta-anneal 16 $B" logs/shotfix2/run.sh C23b 10 12 14 16
echo ALLDONE

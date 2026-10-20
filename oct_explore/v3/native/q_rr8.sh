#!/bin/bash
cd /workspace/qumode_sep/oct_explore
while pgrep -f "tag B32_RR4_S" >/dev/null; do sleep 60; done
CORES=0-4 exec v3/native/gq_bias.sh B32_RR8_t05 --rr 8 --tailw 0.5 --train 10 15 --test 15 20 --proj chain --assign mincut --bias 1 --bias_mlp 1 --es_metric S --npool 800 --bs 2 --steps 2500 --es 250 --nval 60 --warm "" --theta_from results/v3/native/gnnw/B32_b90.json

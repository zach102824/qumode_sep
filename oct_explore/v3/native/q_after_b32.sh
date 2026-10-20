#!/bin/bash
# wait for B32_RR4 to finish, then launch KD16_marg (memory headroom policy)
cd /workspace/qumode_sep/oct_explore
while pgrep -f "tag B32_RR4" >/dev/null; do sleep 60; done
CORES=3 MEMMB=2000 exec v3/native/gq_bias.sh KD16_marg --kd 1.0 --kd_mode marg --teacher results/v3/native/gnnw/A16_full.json --theta_from results/v3/native/gnnw/A16_full.json --train 8 12 --val_n 16 --test 16 --proj chain --assign mincut --bias 1 --bias_mlp 1 --tailw 0.5 --es_metric S --npool 2500 --bs 16 --steps 2500 --es 250 --nval 60 --warm ""

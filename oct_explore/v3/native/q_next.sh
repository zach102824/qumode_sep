#!/bin/bash
cd /workspace/qumode_sep/oct_explore
# wait until fewer than 4 of our python jobs (leave room for F32/mixcal)
while [ "$(pgrep -f 'python3.13 v3/native' | wc -l)" -ge 5 ]; do sleep 60; done
v3/native/gq_bias.sh B16_chain_mc --train 8 12 --test 8 12 16 --proj chain --assign mincut --bias 1 --npool 2500 --x0 results/v3/native/fixed/F16_gray_chain_mc.json
while [ "$(pgrep -f 'python3.13 v3/native' | wc -l)" -ge 5 ]; do sleep 60; done
v3/native/gq.sh G16_local_w --train 8 12 --test 8 12 16 --proj local --assign block --npool 2500 --x0 results/v3/native/fixed/F16_gray_chain.json
while [ "$(pgrep -f 'python3.13 v3/native' | wc -l)" -ge 5 ]; do sleep 60; done
v3/native/gq.sh G16_chain_rand --train 8 12 --test 8 12 16 --proj chain --assign random --npool 2500 --x0 results/v3/native/fixed/F16_gray_chain.json

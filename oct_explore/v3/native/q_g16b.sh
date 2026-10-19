#!/bin/bash
cd /workspace/qumode_sep/oct_explore
while pgrep -f run_evn16.sh >/dev/null; do sleep 30; done
v3/native/gq.sh G16_chain_w --train 8 12 --test 8 12 16 --proj chain --assign block --x0 results/v3/native/fixed/F16_gray_chain.json
v3/native/gq.sh G16_full_w --train 8 12 --test 8 12 16 --proj full --x0 results/v3/native/fixed/F16_qaoa.json

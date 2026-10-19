#!/bin/bash
cd /workspace/qumode_sep/oct_explore
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1
while ! python3.13 -c "import json; d=json.load(open('results/v3/native/gnnw/B20_train.json')); assert '20' in d.get('test',{})" 2>/dev/null; do sleep 60; done
taskset -c 0-6 /usr/bin/python3.13 v3/native/ensemble_eval.py --src results/v3/native/gnnw/B20_train.json --tests 20 --R 32 > logs/v3/native/ens_B20_R32.log 2>&1

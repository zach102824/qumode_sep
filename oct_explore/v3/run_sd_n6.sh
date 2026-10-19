#!/bin/bash
# Oct 19 (Zach ~11:40 CST): SNAP+displacement cavity-only design, n=6 L4 feasibility. GNN-SNAP only (plain dropped by Zach).
# wave 1: spec order (SNAP,D,cost) flat start, 3 inits + QAOA L4 baseline; wave 2: extras (cost-first order x3, vacuum start x1);
# then NF=40 truncation check of every SD run. cores 0-7, python3.13.
cd /workspace/qumode_sep/oct_explore
PY=/usr/bin/python3.13; LG=logs/v3/sd_n6
J() { local th=$1; shift; taskset -c 0-7 env XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=$th XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$th" $PY v3/sd_n6.py "$@"; }
date; echo WAVE1
for k in 0 1 2; do J 2 --tag SD_sdc_flat_i$k --init $k > $LG/SD_sdc_flat_i$k.log 2>&1 & done
J 2 --tag QAOA_n6_L4 --qaoa > $LG/QAOA_n6_L4.log 2>&1 &
wait
date; echo WAVE2
for k in 0 1 2; do J 2 --tag SD_csd_flat_i$k --order csd --init $k > $LG/SD_csd_flat_i$k.log 2>&1 & done
J 2 --tag SD_sdc_vac_i0 --start vac --init 0 > $LG/SD_sdc_vac_i0.log 2>&1 &
wait
date; echo NFCHECK
for t in SD_sdc_flat_i0 SD_sdc_flat_i1 SD_sdc_flat_i2 SD_csd_flat_i0 SD_csd_flat_i1 SD_csd_flat_i2 SD_sdc_vac_i0; do
  o=$(python3.13 -c "import json;print(json.load(open('results/v3/sd_n6/$t.json'))['args']['order'])"); s=$(python3.13 -c "import json;print(json.load(open('results/v3/sd_n6/$t.json'))['args']['start'])")
  J 8 --tag $t --order $o --start $s --eval_from results/v3/sd_n6/$t.json --NF 40 >> $LG/nfcheck.log 2>&1
done
date; echo SDALLDONE

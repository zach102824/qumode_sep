#!/bin/bash
# n=20 explore-only layout-noise test (Zach 2026-10-13). Strict n=20 recipe, relabel skipped (QM_EXPLORE_ONLY=1), inst 5 only.
# Stop: touch logs/3cav/q_n20_noise_explore.STOP
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1 QM_SA_COUNT=raw QM_FIXED_COST=1 QM_EXPLORE_ONLY=1 QM_INSTS=5
STOPF=logs/3cav/q_n20_noise_explore.STOP; D=logs/3cav/n20noise; OUT=noiseless/results/relayout_runs/n20_noise_explore
mkdir -p "$D" "$OUT"
for nz in 0.15 0.3 0.5 1.0 2.0; do
  [ -f "$STOPF" ] && { echo STOP; exit 0; }
  echo "[$(TZ=Asia/Shanghai date +%T) CST] START nz$nz"
  python noiseless/run_relayout_kbit.py --L 4 --s 50 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 1 \
    --explore-mask sa --samode fixed_low --polt 1 --R 1 --trials 1 --outdir $OUT --device 3cav --r 100 --xL 3 --xs 35 \
    --n 20 --ks 6,6,5 --explore 8 --sab 200 --rawk 40 --sanoise $nz > $D/nz$nz.json 2>&1 &
done
wait; echo "[$(TZ=Asia/Shanghai date +%T) CST] ALL DONE"; touch logs/3cav/q_n20_noise_explore.done

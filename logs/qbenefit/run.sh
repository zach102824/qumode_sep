#!/bin/bash
# QUANTUM_BENEFIT.md screens, exact cost. usage: run.sh VARIANT n...   (VARIANT env: NOCLS / RANDREAD / BASE / custom via EXTRA,ENVS)
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1 QM_SA_COUNT=raw QM_FIXED_COST=1 QM_CLS_ALGO=satlike
V=$1; shift
case $V in
 BASE) ;;
 NOCLS) export QM_NOCLS=1;;
 RANDREAD) export QM_RANDREAD=1;;
 BOTH) export QM_NOCLS=1 QM_RANDREAD=1;;
 *) eval "export $ENVS";;
esac
OUT=noiseless/results/relayout_runs/qbenefit/$V; D=logs/qbenefit; mkdir -p $OUT
CM="--L 4 --s 50 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers ${W:-8} --explore-mask sa --samode fixed_low --polt 1 --R ${RR:-1} --trials 1 --sanoise 0.3 --outdir $OUT $EXTRA"
C2="$CM --r 150 --xL 2"; C3="$CM --device 3cav --r 150 --xL 3 --xs 35 --ecd-fast"
declare -A A
A[8]="$C2 --n 8 --xs 25 --explore ${EX:-2}"; A[10]="$C2 --n 10 --xs 35 --explore ${EX:-2}"; A[12]="$C2 --n 12 --xs 50 --explore ${EX:-2}"
A[14]="$C3 --n 14 --ks 4,4,3 --explore ${EX:-2}"; A[16]="$C3 --n 16 --ks 5,4,4 --explore ${EX16:-4}"
for n in "$@"; do
  echo "[$(date '+%F %T') CST] START $V n$n"
  python noiseless/run_relayout_kbit.py ${A[$n]} --sab $((3*n)) --rawk $((2*n)) > $D/${V}_n$n.json 2>&1
  echo "[$(date '+%F %T') CST] DONE $V n$n rc=$?"
done

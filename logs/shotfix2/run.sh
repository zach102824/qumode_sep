#!/bin/bash
# R1-R3 finite-shot screens (FINITE_SHOT_FIXES.md). lowE + CRN always on. usage: run.sh VARIANT n...
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1 QM_SA_COUNT=raw QM_FIXED_COST=1 QM_CLS_ALGO=satlike
V=$1; shift
B="--shot-readout lowE --shot-crn"
case $V in
 B100) F="--shots 100 $B";;
 R1)   F="--shots 100 $B --xbeta 1";;
 R2)   F="--shots 1000 --shot-s0 50 $B";;
 R3)   F="--shots 100 $B --shot-eta-anneal 16";;
 *)    F="$EXTRA";;
esac
OUT=noiseless/results/relayout_runs/shotfix2/$V; D=logs/shotfix2; mkdir -p $OUT
CM="--L 4 --s 50 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers ${W:-8} --explore-mask sa --samode fixed_low --polt 1 --R 1 --trials 1 --sanoise 0.3 --outdir $OUT $F"
C2="$CM --r 150 --xL 2"; C3="$CM --device 3cav --r 150 --xL 3 --xs 35 --ecd-fast"
declare -A A
A[10]="$C2 --n 10 --xs 35 --explore 2"; A[12]="$C2 --n 12 --xs 50 --explore 2"
A[8]="$C2 --n 8 --xs 25 --explore 2"
A[18]="$C3 --n 18 --ks 5,5,5 --explore 6"
A[14]="$C3 --n 14 --ks 4,4,3 --explore 2"; A[16]="$C3 --n 16 --ks 5,4,4 --explore 4"
for n in "$@"; do
  echo "[$(date '+%F %T') CST] START $V n$n"
  python noiseless/run_relayout_kbit.py ${A[$n]} --sab $((3*n)) --rawk $((2*n)) ${POST} > $D/${V}_n$n.json 2>&1
  echo "[$(date '+%F %T') CST] DONE $V n$n rc=$?"
done

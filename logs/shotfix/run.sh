#!/bin/bash
# sl3 recipe + --shots S with fix flags (FINITE_SHOT_FIXES.md). usage: run.sh VARIANT S n...
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1 QM_SA_COUNT=raw QM_FIXED_COST=1 QM_CLS_ALGO=satlike
V=$1; S=$2; shift 2
case $V in
 V0) F="";; V1) F="--shot-readout lowE";; V2) F="--shot-readout lowE --shot-crn";;
 V3) F="--shot-readout lowE --shot-crn --shot-cost cvar --cvar-alpha 0.1";;
esac
OUT=noiseless/results/relayout_runs/shotfix/${V}_S$S; D=logs/shotfix; mkdir -p $OUT
CM="--L 4 --s 50 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 6 --explore-mask sa --samode fixed_low --polt 1 --R 1 --trials 1 --sanoise 0.3 --outdir $OUT --shots $S $F"
C2="$CM --r 150 --xL 2"
declare -A A
A[8]="$C2 --n 8 --xs 25 --explore 2"; A[10]="$C2 --n 10 --xs 35 --explore 2"
for n in "$@"; do
  echo "[$(date '+%F %T') CST] START $V S$S n$n"
  python noiseless/run_relayout_kbit.py ${A[$n]} --sab $((3*n)) --rawk $((2*n)) > $D/${V}_S${S}_n$n.json 2>&1
  echo "[$(date '+%F %T') CST] DONE $V S$S n$n rc=$?"
done

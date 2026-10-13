#!/bin/bash
# STRICT recipe sweep K=3n (replaces q_3cav_strict.sh K=2n) (Zach 2026-10-13): SA=10n RAW evaluations (QM_SA_COUNT=raw, 64 probes), K=3n, polt 1, R1, Gray, fixed_low.
# 10-trial screens only (--trials 1). No retuning, no confirms. Stop: touch logs/3cav/q_3cav_strict3n.STOP
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1 QM_SA_COUNT=raw
STOPF=logs/3cav/q_3cav_strict3n.STOP; DONE=logs/3cav/q_3cav_strict3n.done
OUT=noiseless/results/relayout_runs/kbit_strict3n; D=logs/3cav/strict3n; RES=noiseless/results/MIN_LOOKUP.md
mkdir -p "$OUT" "$D"
CM="--L 4 --s 50 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --polt 1 --R 1 --trials 1 --outdir $OUT"
C2="$CM --r 150 --xL 2"; C3="$CM --device 3cav --r 100 --xL 3 --xs 35"
declare -A A CH XL XS RR EE
A[8]="$C2 --n 8 --xs 25 --explore 1";   CH[8]="2cav 2+3/3";  XL[8]=2; XS[8]=25; RR[8]=150; EE[8]=1
A[10]="$C2 --n 10 --xs 35 --explore 1"; CH[10]="2cav 2+4/4"; XL[10]=2; XS[10]=35; RR[10]=150; EE[10]=1
A[12]="$C2 --n 12 --xs 50 --explore 1"; CH[12]="2cav 2+5/5"; XL[12]=2; XS[12]=50; RR[12]=150; EE[12]=1
A[14]="$C3 --n 14 --ks 4,4,3 --explore 2"; CH[14]="3cav 3+4/4/3"; XL[14]=3; XS[14]=35; RR[14]=100; EE[14]=2
A[16]="$C3 --n 16 --ks 5,4,4 --explore 4"; CH[16]="3cav 3+5/4/4"; XL[16]=3; XS[16]=35; RR[16]=100; EE[16]=4
A[18]="$C3 --n 18 --ks 5,5,5 --explore 6"; CH[18]="3cav 3+5/5/5"; XL[18]=3; XS[18]=35; RR[18]=100; EE[18]=6
A[20]="$C3 --n 20 --ks 6,6,5 --explore 8"; CH[20]="3cav 3+6/6/5"; XL[20]=3; XS[20]=35; RR[20]=100; EE[20]=8
[ -z "$NOHDR" ] && { echo ""; echo "## Strict recipe (SA=10n raw, K=3n, polt 1, R1), 10-trial"; echo ""
  echo "Started $(TZ=Asia/Shanghai date '+%F %H:%M') CST. SA budget = 10n cost evaluations counted raw (QM_SA_COUNT=raw: 64 random probes + anneal steps, repeats count), K = 3n top readouts per explore run, plus the relabel round also reads its own top K, so readout lookups = (E+1)*K (kept deliberately), polt 1 (n single-bit neighbours per new fix-up center), one relabel round, Gray code, fixed_low, 1 trial per instance x 10 instances. Raw lookups = SA + readouts + fix-ups as counted by the code; distinct = unique strings. Eval formula E*S*(2s+1)+(2r+1) with S = xL, s = xs."; echo ""
  echo "| n | chip | E | xL | xs | r | success | mean p(GS) | evals | formula | SA lookups | readout lookups | fix-up lookups | total raw lookups | distinct lookups | share of 2^n (raw) |"
  echo "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"; } >> "$RES"
for n in ${NS:-8 10 12 14 16 18 20}; do
  [ -f "$STOPF" ] && { echo "[$(date +%T)] STOP"; exit 0; }
  f=$D/strict_n$n.json; echo "[$(TZ=Asia/Shanghai date +%T) CST] START n$n"
  # shellcheck disable=SC2086
  python noiseless/run_relayout_kbit.py ${A[$n]} --sab $((10*n)) --rawk $((3*n)) > "$f" 2>&1
  python - "$f" "$n" "${CH[$n]}" "${EE[$n]}" "${XL[$n]}" "${XS[$n]}" "${RR[$n]}" "$RES" "$OUT" <<'PYX'
import json,sys
f,n,ch,E,S,s,r,res,od=sys.argv[1:]; n,E,S,s,r=map(int,(n,E,S,s,r))
t=open(f).read()
try:
    d=json.loads(t[t.index("{"):t.rindex("}")+1]); recs=[json.loads(l) for l in open(f"{od}/{d['tag']}.jsonl") if l.strip()]
except Exception as e:
    open(res,"a").write(f"| {n} | {ch} | {E} | {S} | {s} | {r} | ERROR {type(e).__name__} |||||||||\n"); raise SystemExit
m=lambda k: sum(x.get(k,0) for x in recs)/len(recs)
fe=E*S*(2*s+1)+(2*r+1); ev=d['evals']
ok="ok" if abs(ev-fe)<0.5 else f"MISMATCH"
tot=m('n_lookups_raw')
open(res,"a").write(f"| {n} | {ch} | {E} | {S} | {s} | {r} | {d['success']:.2f} | {d['mean_p_gs']:.3f} | {ev:.0f} | {E}*{S}*{2*s+1}+{2*r+1}={fe} {ok} | {m('n_lookups_sa_raw'):g} | {m('n_lookups_readout_raw'):g} | {m('n_lookups_fixup_raw'):g} | {tot:g} | {m('n_lookups_distinct'):g} | {100*tot/2**n:.3g}% |\n")
PYX
  echo "[$(TZ=Asia/Shanghai date +%T) CST] DONE n$n"
  git add "$RES" && git commit -qm "MIN_LOOKUP: strict recipe K=3n n=$n" && git push -q origin HEAD || true
done
git add logs/3cav/q_3cav_strict3n.sh && git commit -qm "strict recipe K=3n queue" && git push -q origin HEAD || true
touch "$DONE"; echo "[$(TZ=Asia/Shanghai date +%T) CST] ALL DONE"

#!/bin/bash
# FINAL strict recipe (replaces K=3n queue) (Zach 2026-10-13): SA=10n RAW evaluations (QM_SA_COUNT=raw, 64 probes), K=2n, polt 1, R1, Gray, fixed_low.
# 10-trial screens only (--trials 1). No retuning, no confirms. Stop: touch logs/3cav/q_3cav_final.STOP
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1 QM_SA_COUNT=raw
STOPF=logs/3cav/q_3cav_final.STOP; DONE=logs/3cav/q_3cav_final.done
OUT=noiseless/results/relayout_runs/kbit_final; D=logs/3cav/final; RES=noiseless/results/MIN_LOOKUP.md
mkdir -p "$OUT" "$D"
CM="--L 4 --s 50 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --polt 1 --R 1 --trials 1 --outdir $OUT"
C2="$CM --r 150 --xL 2"; C3="$CM --device 3cav --r 100 --xL 3 --xs 35"
declare -A A CH XL XS RR EE
A[8]="$C2 --n 8 --xs 25 --explore 2";   CH[8]="2cav 2+3/3";  XL[8]=2; XS[8]=25; RR[8]=150; EE[8]=2
A[10]="$C2 --n 10 --xs 35 --explore 2"; CH[10]="2cav 2+4/4"; XL[10]=2; XS[10]=35; RR[10]=150; EE[10]=2
A[12]="$C2 --n 12 --xs 50 --explore 2"; CH[12]="2cav 2+5/5"; XL[12]=2; XS[12]=50; RR[12]=150; EE[12]=2
A[14]="$C3 --n 14 --ks 4,4,3 --explore 2"; CH[14]="3cav 3+4/4/3"; XL[14]=3; XS[14]=35; RR[14]=100; EE[14]=2
A[16]="$C3 --n 16 --ks 5,4,4 --explore 4"; CH[16]="3cav 3+5/4/4"; XL[16]=3; XS[16]=35; RR[16]=100; EE[16]=4
A[18]="$C3 --n 18 --ks 5,5,5 --explore 6"; CH[18]="3cav 3+5/5/5"; XL[18]=3; XS[18]=35; RR[18]=100; EE[18]=6
A[20]="$C3 --n 20 --ks 6,6,5 --explore 8"; CH[20]="3cav 3+6/6/5"; XL[20]=3; XS[20]=35; RR[20]=100; EE[20]=8
[ -z "$NOHDR" ] && { echo ""; echo "## FINAL strict recipe (SA=10n raw, K=2n, polt 1, R1, E=max(2,...)), 10-trial"; echo ""
  echo "Started $(TZ=Asia/Shanghai date '+%F %H:%M') CST. SA = 10n raw cost evaluations (QM_SA_COUNT=raw), K = 2n top readouts per explore run and for the relabel round, polt 1, one relabel round, Gray, fixed_low, 1 trial x 10 instances. E = 2,2,2,2,4,6,8 for n=8..20. 2cav (n<=12): xL2, xs 25/35/50, r150; 3cav (n>=14): xL3, xs35, r100. L4 s50 lr0.05 xeta128 reta2."; echo ""
  echo "Formulas: lookups = 10n + (E+1)*2n + E*n (upper bound; fix-ups are not repeated for a center already done). evals = E*S*(2s+1)+(2r+1), S = xL, s = xs. n=14/16/18 reused from kbit_strict (identical recipe)."; echo ""
  echo "| n | chip | E | xL | xs | r | success | mean p(GS) | evals | formula | SA lookups | readout lookups | fix-up lookups | total raw lookups | distinct lookups | share of 2^n (raw) | source |"
  echo "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
  grep -A14 "^## Strict recipe (SA=10n raw, K=2n" "$RES" | grep -E "^\| 1[468] \|" | sed 's/$/ kbit_strict |/'; } >> "$RES"
for n in ${NS:-8 10 12 20}; do
  [ -f "$STOPF" ] && { echo "[$(date +%T)] STOP"; exit 0; }
  f=$D/strict_n$n.json; echo "[$(TZ=Asia/Shanghai date +%T) CST] START n$n"
  # shellcheck disable=SC2086
  python noiseless/run_relayout_kbit.py ${A[$n]} --sab $((10*n)) --rawk $((2*n)) > "$f" 2>&1
  python - "$f" "$n" "${CH[$n]}" "${EE[$n]}" "${XL[$n]}" "${XS[$n]}" "${RR[$n]}" "$RES" "$OUT" <<'PYX'
import json,sys
f,n,ch,E,S,s,r,res,od=sys.argv[1:]; n,E,S,s,r=map(int,(n,E,S,s,r))
t=open(f).read()
try:
    d=json.loads(t[t.index("{"):t.rindex("}")+1]); recs=[json.loads(l) for l in open(f"{od}/{d['tag']}.jsonl") if l.strip()]
except Exception as e:
    open(res,"a").write(f"| {n} | {ch} | {E} | {S} | {s} | {r} | ERROR {type(e).__name__} |||||||||| kbit_final |\n"); raise SystemExit
m=lambda k: sum(x.get(k,0) for x in recs)/len(recs)
fe=E*S*(2*s+1)+(2*r+1); ev=d['evals']
ok="ok" if abs(ev-fe)<0.5 else f"MISMATCH"
tot=m('n_lookups_raw')
open(res,"a").write(f"| {n} | {ch} | {E} | {S} | {s} | {r} | {d['success']:.2f} | {d['mean_p_gs']:.3f} | {ev:.0f} | {E}*{S}*{2*s+1}+{2*r+1}={fe} {ok} | {m('n_lookups_sa_raw'):g} | {m('n_lookups_readout_raw'):g} | {m('n_lookups_fixup_raw'):g} | {tot:g} | {m('n_lookups_distinct'):g} | {100*tot/2**n:.3g}% | kbit_final |\n")
PYX
  python - "$RES" <<'PYS'
import sys
p=sys.argv[1]; L=open(p).read().split("\n"); i=max(k for k,l in enumerate(L) if l.startswith("## FINAL strict recipe"))
j=next(k for k in range(i,len(L)) if L[k].startswith("|---"))+1; e=j
while e<len(L) and L[e].startswith("| "): e+=1
L[j:e]=sorted(L[j:e],key=lambda l:int(l.split("|")[1])); open(p,"w").write("\n".join(L))
PYS
  echo "[$(TZ=Asia/Shanghai date +%T) CST] DONE n$n"
  git add "$RES" && git commit -qm "MIN_LOOKUP: FINAL strict recipe n=$n" && git push -q origin HEAD || true
done
git add logs/3cav/q_3cav_final.sh && git commit -qm "FINAL strict recipe queue" && git push -q origin HEAD || true
touch "$DONE"; echo "[$(TZ=Asia/Shanghai date +%T) CST] ALL DONE"

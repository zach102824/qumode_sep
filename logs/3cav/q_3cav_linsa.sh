#!/bin/bash
# Consistent linear-SA recipe (Zach 2026-10-13). 10-trial screens only (--trials 1).
# Recipe at every n: SA budget sab=9n distinct lookups (QM_SA_PROBES=4n random probes + rest in anneal chains),
# then E explore, top-K readouts with K=n (rule), polt 1, one relabel round. Gray, fixed_low.
# If K=n fails: climb K = 1.5n, 2n, 3n (rounded). If passes: try K = n/2 once to see slack.
# Stop: touch logs/3cav/q_3cav_linsa.STOP. Never creates RESUME_CONFIRMS.
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1
STOPF=logs/3cav/q_3cav_linsa.STOP; DONE=logs/3cav/q_3cav_linsa.done
OUT=noiseless/results/relayout_runs/kbit_linsa; D=logs/3cav/linsa; RES=noiseless/results/MIN_LOOKUP.md; PBAR=0.89
mkdir -p "$OUT" "$D"
C3="--device 3cav --L 4 --s 50 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 8 --samode fixed_low --xL 3 --xs 35 --r 100 --R 1"
C2="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 8 --samode fixed_low --xL 2 --explore 1"
declare -A BASE
BASE[8]="$C2 --n 8 --xs 25"; BASE[10]="$C2 --n 10 --xs 35"; BASE[12]="$C2 --n 12 --xs 50"
BASE[14]="$C3 --n 14 --ks 4,4,3 --explore 2"; BASE[16]="$C3 --n 16 --ks 5,4,4 --explore 4"; BASE[18]="$C3 --n 18 --ks 5,5,5 --explore 6"
declare -A EE=([8]=1 [10]=1 [12]=1 [14]=2 [16]=4 [18]=6)
stopchk(){ [ -f "$STOPF" ] && echo "[$(date +%T)] STOP" && exit 0; }
screen(){ # n k -> PASS/FAIL
  local n=$1 k=$2 sab=$((9*$1)) name="linsa_n$1_k$2" f; f=$D/$name.json; stopchk
  echo "[$(date +%T)] START $name" >&2
  # shellcheck disable=SC2086
  QM_SA_PROBES=$((4*n)) python noiseless/run_relayout_kbit.py ${BASE[$n]} --sab $sab --rawk $k --polt 1 --trials 1 --outdir $OUT > "$f" 2>&1
  python - "$f" "$n" "$k" "$sab" "${EE[$n]}" "$RES" "$OUT" "$PBAR" <<'PYX'
import json,sys
f,n,k,sab,E,res,od,pbar=sys.argv[1:]; n=int(n)
t=open(f).read()
try: d=json.loads(t[t.index("{"):t.rindex("}")+1])
except Exception:
    open(res,"a").write(f"| {n} | ERROR | {sab} | {k} | {E} ||||||\n"); print("FAIL"); raise SystemExit
recs=[json.loads(l) for l in open(f"{od}/{d['tag']}.jsonl") if l.strip()]
m=lambda key: sum(r.get(key,0) for r in recs)/max(len(recs),1)
s,pp=d["success"],d["mean_p_gs"]; ok=s>=0.9 and pp>=float(pbar); L=m('n_lookups_distinct')
open(res,"a").write(f"| {n} | {L:.1f} | {m('n_lookups_sa'):.1f} | {k} | {E} | {s:.2f} | {pp:.3f} | {d['evals']:.0f} | {100*L/2**n:.3g}% | {'PASS' if ok else 'fail'} |\n")
print("PASS" if ok else "FAIL")
PYX
}
{ echo ""; echo "## Consistent linear-SA recipe (10-trial screens, started $(date '+%F %H:%M') CST)"; echo ""
  echo "Same steps at every n: SA with sab = 9n distinct lookups (QM_SA_PROBES = 4n random probes, rest in anneal chains of 8n steps), E explore runs (1,1,1,2,4,6), top-K readouts with K = n, polt 1 single-bit fix-ups, one relabel round (r 150 n<=12 on 2 cav, r 100 n>=14 on 3 cav), Gray code, fixed_low. If K = n fails, K steps up 1.5n, 2n, 3n; if it passes, K = n/2 is tried once. Pass = success >= 0.9 and mean p(GS) >= $PBAR."; echo ""
  echo "| n | lookups total | SA lookups | K | E | success | mean p(GS) | evals | share of 2^n | result |"
  echo "|---|---|---|---|---|---|---|---|---|---|"; } >> "$RES"
for n in 8 10 12 14 16 18; do
  if [ "$(screen $n $n)" = PASS ]; then screen $n $((n/2)) >/dev/null
  else for k in $((3*n/2)) $((2*n)) $((3*n)); do [ "$(screen $n $k)" = PASS ] && break; done; fi
  git add "$RES" && git commit -qm "MIN_LOOKUP: linear-SA recipe n=$n" && git push -q origin HEAD || true
done
python - "$RES" <<'PYF'
import re,sys,numpy as np
res=sys.argv[1]; t=open(res).read(); sec=t[t.rindex("## Consistent linear-SA recipe"):]
best={}
for l in sec.splitlines():
    c=[x.strip() for x in l.strip("|").split("|")]
    if len(c)==10 and c[-1]=="PASS":
        n=int(c[0]); L=float(c[1]); best[n]=min(best.get(n,1e9),L)
if len(best)>=2:
    ns=np.array(sorted(best)); Ls=np.array([best[n] for n in ns]); a,b=np.polyfit(ns,Ls,1)
    r2=1-((Ls-(a*ns+b))**2).sum()/((Ls-Ls.mean())**2).sum()
    open(res,"a").write(f"\nLinear fit of cheapest passing total lookups vs n (linear-SA recipe): total ~ {a:.1f} n {b:+.1f} (R^2 {r2:.2f}); points "+", ".join(f"n{n}:{best[n]:.0f}" for n in ns)+"\n")
PYF
git add "$RES" logs/3cav/q_3cav_linsa.sh && git commit -qm "MIN_LOOKUP: linear-SA recipe table + fit" && git push -q origin HEAD || true
touch "$DONE"; echo "[$(date +%T)] ALL DONE"

#!/bin/bash
# No-SA sweep (Zach 2026-10-13): 10-trial screens ONLY (--trials 1), no 50-trial confirms.
# Pass = success >= 0.9 and mean p(GS) >= 0.89. explore-mask zero & random, sab 0.
# Per n: start at table K, descend ladder to smallest pass (or climb if start fails), then polt 0 at that K.
# n=8-12 use the 2-cavity base (fixes cutnext4/5 n12 ERROR: they passed 3cav ks 4,3,3 for n=12).
# Never creates RESUME_CONFIRMS. Stop: touch logs/3cav/q_3cav_nosa.STOP.
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1
STOPF=logs/3cav/q_3cav_nosa.STOP; DONE=logs/3cav/q_3cav_nosa.done
OUT=noiseless/results/relayout_runs/kbit_cutnext5   # same dir => identical configs from cutnext5 resume instantly
D=logs/3cav/nosa; RES=noiseless/results/MIN_LOOKUP.md; PBAR=0.89
mkdir -p "$OUT" "$D"
C3="--device 3cav --L 4 --s 50 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 8 --samode fixed_low --xL 3 --xs 35 --r 100 --R 1"
C2="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 8 --samode fixed_low --xL 2 --explore 1"
declare -A BASE K0
BASE[8]="$C2 --n 8 --xs 25";  K0[8]=9
BASE[10]="$C2 --n 10 --xs 35"; K0[10]=37
BASE[12]="$C2 --n 12 --xs 50"; K0[12]=23
BASE[14]="$C3 --n 14 --ks 4,4,3 --explore 2"; K0[14]=2
BASE[16]="$C3 --n 16 --ks 5,4,4 --explore 4"; K0[16]=10
BASE[18]="$C3 --n 18 --ks 5,5,5 --explore 6"; K0[18]=12
LADDER="1 2 3 4 6 8 10 12 16 24 32 48"
stopchk(){ [ -f "$STOPF" ] && echo "[$(date +%T)] STOP" && exit 0; }
while [ ! -f logs/3cav/q_3cav_cutnext5.done ] || pgrep -f 'run_relayout_kbit.py' >/dev/null 2>&1; do stopchk; sleep 20; done

screen(){ # n mask k p -> PASS/FAIL
  local n=$1 mask=$2 k=$3 p=$4 name="nosa_n$1_$2_k$3_p$4" f
  f=$D/$name.json; stopchk
  echo "[$(date +%T)] START $name" >&2
  # shellcheck disable=SC2086
  python noiseless/run_relayout_kbit.py ${BASE[$n]} --explore-mask $mask --sab 0 --rawk $k --polt $p --trials 1 --outdir $OUT > "$f" 2>&1
  python - "$f" "$name" "$mask" "$k" "$p" "$RES" "$OUT" "$PBAR" <<'PYX'
import json,sys
f,name,mask,k,p,res,od,pbar=sys.argv[1:]
t=open(f).read()
try: d=json.loads(t[t.index("{"):t.rindex("}")+1])
except Exception:
    open(res,"a").write(f"| {name} | {mask} | 0 | {k} | {p} | ERROR ||||||\n"); print("FAIL"); raise SystemExit
recs=[json.loads(l) for l in open(f"{od}/{d['tag']}.jsonl") if l.strip()]
m=lambda key: sum(r.get(key,0) for r in recs)/max(len(recs),1)
s,pp=d["success"],d["mean_p_gs"]; ok=s>=0.9 and pp>=float(pbar)
open(res,"a").write(f"| {name} | {mask} | 0 | {k} | {p} | {d['n_trials']} | {s:.2f} | {pp:.3f} | {d['evals']:.0f} | "
  f"{m('n_lookups_distinct'):.1f} | {m('n_lookups_sa'):.1f} | {'PASS' if ok else 'fail'} |\n")
print("PASS" if ok else "FAIL")
PYX
}

{ echo ""; echo "## No-SA sweep, 10-trial screens only (auto, started $(date '+%F %H:%M') CST)"; echo ""
  echo "Pass = success >= 0.9 and mean p(GS) >= $PBAR. sab 0 (no SA lookups); K descended to smallest pass, then polt 0."; echo ""
  echo "| run | mask | sab | rawk K | polt | trials | success | mean p(GS) | evals | total lookups | SA lookups | result |"
  echo "|---|---|---|---|---|---|---|---|---|---|---|---|"; } >> "$RES"

for n in 8 10 12 14 16 18; do
  for mask in zero random; do
    k0=${K0[$n]}; best=""
    if [ "$(screen $n $mask $k0 1)" = PASS ]; then
      best=$k0
      for k in $(echo $LADDER | tr ' ' '\n' | sort -rn); do
        [ "$k" -ge "$k0" ] && continue
        if [ "$(screen $n $mask $k 1)" = PASS ]; then best=$k; else break; fi
      done
    else
      for k in $LADDER; do
        [ "$k" -le "$k0" ] && continue
        if [ "$(screen $n $mask $k 1)" = PASS ]; then best=$k; break; fi
      done
    fi
    echo "[$(date +%T)] n$n $mask smallest passing K = ${best:-none}"
    [ -n "$best" ] && echo "[$(date +%T)] n$n $mask K$best polt0 -> $(screen $n $mask $best 0)"
  done
  git add "$RES" && git commit -qm "MIN_LOOKUP: no-SA sweep n=$n (10-trial)" && git push -q origin HEAD || true
done

python noiseless/analysis/final_10trial_table.py "$RES" $PBAR "$RES" noiseless/results/MIN_LOOKUP_RULE.md
git add "$RES" noiseless/analysis/final_10trial_table.py && git commit -qm "MIN_LOOKUP: final 10-trial table + refit" && git push -q origin HEAD || true
touch "$DONE"; echo "[$(date +%T)] ALL DONE"

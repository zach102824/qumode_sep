#!/bin/bash
# MIN_LOOKUP_RULE validation: diag pass -> fit K(n), polt(n) -> 10-trial rule screens -> 50-trial confirms.
# Gated on q_3cav_minlookup.done. Own outdir (kbit_rule) so no checkpoint clashes. Stop: touch logs/3cav/q_3cav_rule.STOP
cd /workspace/qumode_sep
source /workspace/venv-qumode/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1
STOPF=logs/3cav/q_3cav_rule.STOP; DONE=logs/3cav/q_3cav_rule.done
OUT=noiseless/results/relayout_runs/kbit_rule; DIAG=$OUT/diag; mkdir -p $DIAG logs/3cav/rule
RES=noiseless/results/MIN_LOOKUP_RULE.md
C3="--device 3cav --L 4 --s 50 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --xL 3 --xs 35 --r 100 --R 1"
C2="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --xL 2 --explore 1"
declare -A BASE SAB0
BASE[18]="$C3 --n 18 --ks 5,5,5 --explore 4"; SAB0[18]=162
BASE[16]="$C3 --n 16 --ks 5,4,4 --explore 4"; SAB0[16]=128
BASE[14]="$C3 --n 14 --ks 4,4,3 --explore 2"; SAB0[14]=64
BASE[12]="$C2 --n 12 --xs 50"; SAB0[12]=72
BASE[10]="$C2 --n 10 --xs 35"; SAB0[10]=50
BASE[8]="$C2 --n 8 --xs 25";  SAB0[8]=32
stopchk(){ [ -f "$STOPF" ] && echo "[$(date +%T)] STOP" && exit 0; }
echo "[$(date +%T)] rule: waiting for logs/3cav/q_3cav_minlookup.done"
while [ ! -f logs/3cav/q_3cav_minlookup.done ]; do stopchk; sleep 30; done
while pgrep -f 'run_relayout_kbit.py' >/dev/null 2>&1; do stopchk; sleep 10; done
# A. diag pass (base settings, 10 trials, logs GS ranks)
for n in 18 16 14 12 10 8; do stopchk
  echo "[$(date +%T)] diag n=$n"
  python noiseless/run_relayout_kbit.py ${BASE[$n]} --sab ${SAB0[$n]} --rawk 48 --polt 3 --trials 1 --outdir $DIAG > logs/3cav/rule/diag_n$n.json 2>&1
done
# B. fit
python noiseless/analysis/min_lookup_rule_fit.py $OUT/rule_fit.json $DIAG noiseless/results/relayout_runs/kbit > logs/3cav/rule/fit.txt 2>&1
getf(){ python -c "import json;d=json.load(open('$OUT/rule_fit.json'));print(d.get('$1',{}).get('$2',$3))"; }
# C/D. screen name n sab probes rawk polt trials -> PASS/FAIL + row
screen(){
  local name=$1 n=$2 sab=$3 pr=$4 k=$5 p=$6 tr=$7 f=logs/3cav/rule/$1.json
  stopchk; echo "[$(date +%T)] START $name" >&2
  QM_SA_PROBES=$pr python noiseless/run_relayout_kbit.py ${BASE[$n]} --sab $sab --rawk $k --polt $p --trials $tr --outdir $OUT/pr$pr > "$f" 2>&1
  python - "$f" "$name" "$sab" "$pr" "$k" "$p" "$RES" "$OUT/pr$pr" <<'PYX'
import json,sys
f,name,sab,pr,k,p,res,od=sys.argv[1:]
t=open(f).read()
try: d=json.loads(t[t.index("{"):t.rindex("}")+1])
except Exception: print("FAIL"); open(res,"a").write(f"| {name} | {sab} | {pr} | {k} | {p} | ERROR |||||\n"); sys.exit()
recs=[json.loads(l) for l in open(f"{od}/{d['tag']}.jsonl") if l.strip()]
m=lambda key: sum(r.get(key,0) for r in recs)/max(len(recs),1)
s,pp=d["success"],d["mean_p_gs"]; ok=s>=0.9 and pp>=0.9
open(res,"a").write(f"| {name} | {sab} | {pr} | {k} | {p} | {d['n_trials']} | {s:.2f} | {pp:.3f} | {m('n_lookups_distinct'):.1f} | {m('n_lookups_sa'):.1f} | {'PASS' if ok else 'fail'} |\n")
print("PASS" if ok else "FAIL")
PYX
}
{ echo ""; echo "## Validation runs (auto-appended by logs/3cav/q_3cav_rule.sh, started $(date '+%F %H:%M') CST)"; echo "";
  echo "Fit (from diag pass + earlier rawk-48 records with ranks):"; echo '```'; cat logs/3cav/rule/fit.txt; echo '```'; echo "";
  echo "| run | sab | SA probes | rawk K | polt | trials | success | mean p(GS) | total lookups | SA lookups | result |";
  echo "|---|---|---|---|---|---|---|---|---|---|---|"; } >> $RES
for n in 18 16 14 12 10 8; do
  K=$(getf $n K 48); P=$(getf $n polt 3)
  if [ $n -ge 12 ]; then sab=16; pr=16; else sab=${SAB0[$n]}; pr=64; fi
  r=$(screen n${n}_rule_sab${sab}_k${K}_p${P} $n $sab $pr $K $P 1)
  if [ "$r" != PASS ]; then
    K2=$(( K*2 )); [ $K2 -gt 48 ] && K2=48; sab=$([ $n -ge 12 ] && echo 64 || echo ${SAB0[$n]}); pr=64
    r=$(screen n${n}_rule2_sab${sab}_k${K2}_p${P} $n $sab $pr $K2 $P 1); K=$K2
  fi
  [ "$r" = PASS ] && screen n${n}_confirm_sab${sab}_k${K}_p${P} $n $sab $pr $K $P 5 >/dev/null
done
git add $RES $OUT/rule_fit.json && git commit -m "MIN_LOOKUP_RULE: validation results" && git push origin HEAD
touch $DONE; echo "[$(date +%T)] ALL DONE"

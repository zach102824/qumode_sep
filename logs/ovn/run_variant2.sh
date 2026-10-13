#!/bin/bash
# usage: run_variant.sh NAME SABMUL "NS" [ENV=VAL ...]   (one shared rule for all n; final recipe otherwise)
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate
NAME=$1; SM=$2; NSL=$3; shift 3
export PYTHONPATH=. OMP_NUM_THREADS=1 QM_SA_COUNT=raw QM_FIXED_COST=1 "$@"
OUT=noiseless/results/relayout_runs/ovn_$NAME; D=logs/ovn/$NAME; RES=noiseless/results/OVERNIGHT_IMPROVEMENTS.md
mkdir -p "$OUT" "$D"
CM="--L 4 --s 50 --lr 0.05 --xeta 128 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --polt 1 --R 1 --trials 1 --sanoise 0.3 --outdir $OUT"
RR=${OVN_R:-150}; C2="$CM --r $RR --xL 2"; C3="$CM --device 3cav --r $RR --xL 3 --xs 35"
declare -A A EE XL XS
A[8]="$C2 --n 8 --xs 25 --explore 2"; EE[8]=2; XL[8]=2; XS[8]=25
A[10]="$C2 --n 10 --xs 35 --explore 2"; EE[10]=2; XL[10]=2; XS[10]=35
A[12]="$C2 --n 12 --xs 50 --explore 2"; EE[12]=2; XL[12]=2; XS[12]=50
A[14]="$C3 --n 14 --ks 4,4,3 --explore 2"; EE[14]=2; XL[14]=3; XS[14]=35
A[16]="$C3 --n 16 --ks 5,4,4 --explore 4"; EE[16]=4; XL[16]=3; XS[16]=35
A[18]="$C3 --n 18 --ks 5,5,5 --explore 6"; EE[18]=6; XL[18]=3; XS[18]=35
A[20]="$C3 --n 20 --ks 6,6,5 --explore 8"; EE[20]=8; XL[20]=3; XS[20]=35
for kv in ${OVN_EMAP//,/ }; do k=${kv%%:*}; v=${kv##*:}; EE[$k]=$v; A[$k]="$(echo "${A[$k]}" | sed "s/--explore [0-9]*/--explore $v/")"; done
for n in $NSL; do
  [ -f logs/ovn/STOP ] && { echo STOP; exit 0; }
  f=$D/n$n.json; echo "[$(TZ=Asia/Shanghai date '+%F %T') CST] START $NAME n$n"
  python noiseless/run_relayout_kbit.py ${A[$n]} --sab $((SM*n)) --rawk $((2*n)) > "$f" 2>&1
  python - "$f" "$n" "${EE[$n]}" "${XL[$n]}" "${XS[$n]}" "$SM" "$RES" "$OUT" "$NAME" "$* ${OVN_EMAP:+EMAP=$OVN_EMAP} r=$RR" "$RR" <<'PYX'
import json,sys
f,n,E,S,s,sm,res,od,name,env,rr=sys.argv[1:]; rr=int(rr); n,E,S,s,sm=map(int,(n,E,S,s,sm))
t=open(f).read(); now=__import__('subprocess').run("TZ=Asia/Shanghai date '+%H:%M'",shell=True,capture_output=True,text=True).stdout.strip()
try:
    d=json.loads(t[t.index("{"):t.rindex("}")+1]); recs=[json.loads(l) for l in open(f"{od}/{d['tag']}.jsonl") if l.strip()]
except Exception as e:
    open(res,"a").write(f"| {name} | {n} | ERROR {type(e).__name__} |\n"); raise SystemExit
key=lambda x:(x['n_lookups_sa_raw'],x['n_lookups_readout_raw'],x['n_lookups_fixup_raw'],x['n_lookups_fixed'])
ks={key(x) for x in recs}; exp=sm*n+3*n*(E+1); ev=d['evals']; fe=E*S*(2*s+1)+2*rr+1
mins=min(r.get('p_gs',r.get('p_gs_final',1)) for r in recs) if recs and ('p_gs' in recs[0]) else None
if len(ks)!=1: open(res,"a").write(f"| {name} | {n} | ERROR non-identical lookups {sorted(ks)} |\n"); raise SystemExit
sa,rd,fx,tot=ks.pop()
open(res,"a").write(f"| {name} | {n} | {env} | {len(recs)} | {d['success']:.2f} | {d['mean_p_gs']:.3f} | {ev:.0f}{'' if abs(ev-fe)<0.5 else ' (formula '+str(fe)+')'} | {sa} / {rd} / {fx} | {tot}{'' if tot==exp else ' (EXPECTED '+str(exp)+')'} | {now} |\n")
PYX
  git add "$RES" && git commit -qm "OVERNIGHT: $NAME n=$n" && git push -q origin HEAD || true
done
echo "[$(TZ=Asia/Shanghai date '+%F %T') CST] DONE $NAME"

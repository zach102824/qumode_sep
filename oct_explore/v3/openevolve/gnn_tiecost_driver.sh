#!/bin/bash
# gnn_tiecost NON-STOP driver. Stop: touch runs/gnn_tiecost/gnn_tiecost.STOP
source /workspace/qumode_sep/oct_explore/v3/openevolve/env_gnn_tiecost.sh
OUT=runs/gnn_tiecost; STOP=$OUT/gnn_tiecost.STOP
mkdir -p $OUT/checkpoints logs
export OE_RUN_DIR=$PWD/$OUT
log(){ echo "$(date '+%F %T') $*" | tee -a $OUT/driver.log; }
echo $$ > $OUT/driver.pid; log "driver start pid $$ L=$OE_L"
latest(){ ls -d $OUT/checkpoints/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1; }
fails=0
while [ ! -f $STOP ]; do
  bash gnn_tiecost_services.sh; protect_tiecost
  CK=$(latest)
  if [ -z "$CK" ]; then
    log "no checkpoint: bootstrap"
    python3.13 bootstrap_gnn_tiecost.py >> $OUT/bootstrap.log 2>&1 || { log "bootstrap failed"; sleep 60; continue; }
    CK=$(latest)
  fi
  LAST=${CK##*_}
  # run chunks of 20 iterations, forever
  NITER=20
  log "launch from $CK (iter $LAST) for $NITER iterations"
  t=$(date +%s)
  python3.13 -m openevolve.cli $OUT/seed_program.py net/evaluator_gnn_tiecost.py \
    --config config_gnn_tiecost.yaml --iterations $NITER --output $OUT --checkpoint $CK \
    >> $OUT/evolve.log 2>&1 &
  echo $! > $OUT/cli.pid
  while kill -0 $(cat $OUT/cli.pid) 2>/dev/null; do
    [ -f $STOP ] && { kill $(cat $OUT/cli.pid) 2>/dev/null; log "STOP file: killed cli"; }
    # refresh STATUS.md lightly
    python3.13 - <<'PY' 2>/dev/null || true
import os, time, glob, json
OUT="runs/gnn_tiecost"
lines=[f"# gnn_tiecost STATUS", f"updated: {time.strftime('%F %T %z')}", ""]
alive=os.path.exists(f"{OUT}/cli.pid") and os.path.exists(f"/proc/{open(f'{OUT}/cli.pid').read().strip()}")
lines.append(f"- cli_alive: {alive}")
cks=sorted(glob.glob(f"{OUT}/checkpoints/checkpoint_*"), key=lambda p:int(p.rsplit('_',1)[-1]))
lines.append(f"- checkpoints: {len(cks)} latest={cks[-1] if cks else None}")
if cks:
  info=f"{cks[-1]}/best_program_info.json"
  if os.path.exists(info):
    d=json.load(open(info)); m=d.get("metrics",{})
    lines.append(f"- best_score: {m.get('combined_score')} growth={m.get('growth_proxy')} n6_med={m.get('n6_med')} n8_med={m.get('n8_med')}")
lines.append(f"- stop_file: {os.path.exists(OUT+'/gnn_tiecost.STOP')}")
open(f"{OUT}/STATUS.md","w").write("\n".join(lines)+"\n")
PY
    sleep 30
  done
  wait; rc=$?; log "cli exited rc=$rc after $(( $(date +%s)-t ))s"
  if [ $(( $(date +%s)-t )) -lt 120 ]; then fails=$((fails+1)); sleep $(( fails < 6 ? 30*fails : 600 )); else fails=0; fi
done
log "driver stopped by $STOP"

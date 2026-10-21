#!/bin/bash
# net16r NON-STOP driver. Alternates explore (60 it, config_net16r.yaml) / exploit (40 it, config_net16r_exploit.yaml) forever,
# resuming from the latest checkpoint after any exit/crash. Stop: create runs/net16r/net16r.STOP (only net16r reads it).
source /workspace/qumode_sep/oct_explore/v3/openevolve/env16r.sh
OUT=runs/net16r; STOP=$OUT/net16r.STOP; ST=$OUT/phase.env; export OE_RUN_DIR=$PWD/$OUT
log(){ echo "$(date '+%F %T') $*" >> $OUT/driver.log; }
echo $$ > $OUT/driver.pid; log "driver start pid $$"
[ -f $ST ] || echo "PHASE=explore END=60" > $ST
latest(){ ls -d $OUT/checkpoints/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1; }
fails=0
while [ ! -f $STOP ]; do
  bash net16r_services.sh
  source $ST; CK=$(latest); LAST=${CK##*_}
  if [ -z "$CK" ]; then log "no checkpoint_0: run bootstrap16r.py first"; sleep 600; continue; fi
  if [ "$LAST" -ge "$END" ]; then
    if [ "$PHASE" = explore ]; then PHASE=exploit; END=$((LAST+40)); else PHASE=explore; END=$((LAST+60)); fi
    echo "PHASE=$PHASE END=$END" > $ST; log "phase -> $PHASE until iteration $END"
  fi
  if [ -f $OUT/reseed.pending ]; then python3.13 reseed16r.py $CK >> $OUT/driver.log 2>&1 && rm -f $OUT/reseed.pending; fi
  CFGF=config_net16r.yaml; [ "$PHASE" = exploit ] && CFGF=config_net16r_exploit.yaml
  log "launch $PHASE from $CK (iter $LAST) for $((END-LAST)) iterations"
  t=$(date +%s)
  python3.13 -m openevolve.cli $OUT/seed_program.py net/evaluator_net16r.py --config $CFGF --iterations $((END-LAST)) --output $OUT --checkpoint $CK >> $OUT/evolve.log 2>&1 &
  echo $! > $OUT/cli.pid; while kill -0 $(cat $OUT/cli.pid) 2>/dev/null; do [ -f $STOP ] && { kill $(cat $OUT/cli.pid); log "STOP file: killed cli"; }; sleep 20; done
  wait; rc=$?; log "cli exited rc=$rc after $(( $(date +%s)-t ))s"
  if [ $(( $(date +%s)-t )) -lt 300 ]; then fails=$((fails+1)); sleep $(( fails < 6 ? 60*fails : 1200 )); else fails=0; fi
done
log "driver stopped by $STOP"

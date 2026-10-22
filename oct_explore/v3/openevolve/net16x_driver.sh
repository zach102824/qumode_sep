#!/bin/bash
# net16x NON-STOP driver. PROTECTED phase (config_net16x_protected.yaml, no migration) until iteration 160 (= 40 evals per island x 4 islands),
# then creates runs/net16x/wildcard.open and alternates explore (60 it, config_net16x.yaml) / exploit (40 it, config_net16x_exploit.yaml) forever,
# resuming from the latest checkpoint after any exit/crash. Stop: create runs/net16x/net16x.STOP (only net16x reads it).
source /workspace/qumode_sep/oct_explore/v3/openevolve/env16x.sh
OUT=runs/net16x; STOP=$OUT/net16x.STOP; ST=$OUT/phase.env; export OE_RUN_DIR=$PWD/$OUT
log(){ echo "$(date '+%F %T') $*" >> $OUT/driver.log; }
echo $$ > $OUT/driver.pid; log "driver start pid $$"
[ -f $ST ] || echo "PHASE=protected END=160" > $ST
latest(){ ls -d $OUT/checkpoints/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1; }
fails=0
while [ ! -f $STOP ]; do
  bash net16x_services.sh; protect16x
  source $ST; CK=$(latest); LAST=${CK##*_}
  if [ -z "$CK" ]; then log "no checkpoint_0: run bootstrap16x.py first"; sleep 600; continue; fi
  if [ "$LAST" -ge "$END" ]; then
    if [ "$PHASE" = protected ]; then touch $OUT/wildcard.open; log "protection over: wildcard island open, migration on"; fi
    if [ "$PHASE" = explore ]; then PHASE=exploit; END=$((LAST+40)); else PHASE=explore; END=$((LAST+60)); fi
    echo "PHASE=$PHASE END=$END" > $ST; log "phase -> $PHASE until iteration $END"
  fi
  if [ -f $OUT/reseed.pending ] && [ -f $OUT/wildcard.open ]; then python3.13 reseed16x.py $CK >> $OUT/driver.log 2>&1 && rm -f $OUT/reseed.pending; fi
  CFGF=config_net16x.yaml; [ "$PHASE" = exploit ] && CFGF=config_net16x_exploit.yaml; [ "$PHASE" = protected ] && CFGF=config_net16x_protected.yaml
  log "launch $PHASE from $CK (iter $LAST) for $((END-LAST)) iterations"
  t=$(date +%s)
  python3.13 -m openevolve.cli $OUT/seed_program.py net/evaluator_net16x.py --config $CFGF --iterations $((END-LAST)) --output $OUT --checkpoint $CK >> $OUT/evolve.log 2>&1 &
  echo $! > $OUT/cli.pid; while kill -0 $(cat $OUT/cli.pid) 2>/dev/null; do [ -f $STOP ] && { kill $(cat $OUT/cli.pid); log "STOP file: killed cli"; }; sleep 20; done
  wait; rc=$?; log "cli exited rc=$rc after $(( $(date +%s)-t ))s"
  if [ $(( $(date +%s)-t )) -lt 300 ]; then fails=$((fails+1)); sleep $(( fails < 6 ? 60*fails : 1200 )); else fails=0; fi
done
log "driver stopped by $STOP"

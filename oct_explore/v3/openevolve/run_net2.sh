#!/bin/bash
# usage: CFG=16|32 CORES=5-6 ITER=200 ./run_net2.sh   (seeded from runs/net_smoke best)
cd "$(dirname "$0")"; export PATH=$HOME/.local/bin:$PATH OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
: "${CURSOR_API_KEY:?CURSOR_API_KEY must be set}"
if [ "$CFG" = 16 ]; then export OE_TRAIN_N=8,12 OE_VAL_N=16 OE_CHECK_N=16 OE_NVAL=${OE_NVAL:-60}; OUT=runs/net16; else export OE_TRAIN_N=10,15 OE_VAL_N=20 OE_CHECK_N=20 OE_NVAL=${OE_NVAL:-30}; OUT=runs/net32; fi
export OE_BUDGET=${OE_BUDGET:-180} OE_NVAL=${OE_NVAL:-100}
SEED=${SEED:-net/initial_net$CFG.py}; mkdir -p $OUT; cp $SEED $OUT/seed_program.py
sed -i 's/bs.get(n, 8)/bs.get(n, 8)/; s/bs = {10: 8, 15: 4}/bs = {8: 16, 10: 8, 15: 4}/; s/idx = rng.choice(data\[n\]\["F"\], bs\[n\], replace=False)/idx = rng.choice(data[n]["F"], bs.get(n, 8), replace=False)/' $OUT/seed_program.py
curl -s -m 2 127.0.0.1:8765/v1/models >/dev/null || { setsid nohup python3.13 cursor_adapter.py --port 8765 --model composer-2.5 --conc 3 > logs/adapter.log 2>&1 & sleep 3; }
( while true; do sleep 120; for c in $OUT/checkpoints/checkpoint_*; do [ -f $c/best_program.py ] || continue; k=$(basename $c); mkdir -p $OUT/check; [ -f $OUT/check/$k.json ] && continue
    OE_N20=1 OE_MEM_MB=6000 taskset -c $CORES python3.13 net/evaluator_net.py $c/best_program.py > $OUT/check/$k.json 2> $OUT/check/$k.err; done
    pgrep -f "openevolve.cli.*$OUT" >/dev/null || break; done ) &
taskset -c $CORES python3.13 -m openevolve.cli $OUT/seed_program.py net/evaluator_net.py --config config_net$CFG.yaml --iterations ${ITER:-200} --output $OUT
wait

#!/bin/bash
# Keep improve exploration alive until STOP file exists.
cd /workspace/qumode_sep
STOP=logs/improve/STOP
LOG=logs/improve/watchdog.log
mkdir -p logs/improve
while true; do
  date "+%Y-%m-%d %H:%M:%S CST watchdog tick" >> "$LOG"
  if [ -f "$STOP" ]; then
    echo "STOP present; exiting watchdog" >> "$LOG"
    exit 0
  fi
  # Is any queue or runner active?
  if pgrep -f 'logs/improve/queue[0-9]+\.sh' >/dev/null || pgrep -f 'run_relayout_kbit.py' >/dev/null; then
    echo "compute busy" >> "$LOG"
  else
    echo "IDLE — launching emergency queue" >> "$LOG"
    NEXT=logs/improve/queue_auto_$(date +%s).sh
    cat > "$NEXT" <<'INNER'
#!/bin/bash
cd /workspace/qumode_sep; source /workspace/venv-qumode/bin/activate; export PYTHONPATH=. OMP_NUM_THREADS=1
B="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 16 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --rawk 24 --polt 2 --trials 5"
B32="--L 4 --s 50 --r 150 --R 1 --lr 0.05 --xeta 32 --reta 2 --ham-set wmaxsat --code gray --workers 8 --explore-mask sa --samode fixed_low --rawk 24 --polt 2 --trials 5"
run(){ name=$1; shift; echo "[$(date +%H:%M:%S)] START $name"; python noiseless/run_relayout_kbit.py "$@" > logs/improve/$name.json 2>&1; echo "[$(date +%H:%M:%S)] DONE $name"; }
# Prefer unexplored cheap points across the ladder
run n16_E22_sab128_rawk24_xs35_R1_c50 --n 16 --explore 22 --sab 128 $B --xs 35
run n16_E24_sab128_rawk24_xs35_R1_c50 --n 16 --explore 24 --sab 128 $B --xs 35
run n16_E26_sab128_rawk24_xs35_R1_c50 --n 16 --explore 26 --sab 128 $B --xs 35
run n14_E3_sab98_rawk24_xs35_R1_xeta32_c50 --n 14 --explore 3 --sab 98 $B32 --xs 35
run n14_E9_sab98_rawk24_xs35_R1_xeta32_c50 --n 14 --explore 9 --sab 98 $B32 --xs 35
run n12_E2_sab72_rawk32_xL2_R1_c50 --n 12 --explore 2 --sab 72 $B --xL 2 --rawk 32
run n08_E1_sab32_rawk32_xL2_R1_c50 --n 8 --explore 1 --sab 32 $B --xL 2 --rawk 32
run n10_E1_sab50_rawk32_xL2_R1_c50 --n 10 --explore 1 --sab 50 $B --xL 2 --rawk 32
run n16_E12_sab128_rawk24_xs35_R1_xeta32_c50 --n 16 --explore 12 --sab 128 $B32 --xs 35
run n16_E14_sab128_rawk24_xs35_R1_xeta32_c50 --n 16 --explore 14 --sab 128 $B32 --xs 35
INNER
    chmod +x "$NEXT"
    setsid nohup bash -c "$NEXT >> ${NEXT%.sh}.out 2>&1" </dev/null >/dev/null 2>&1 &
    echo "started $NEXT pid $!" >> "$LOG"
  fi
  sleep 1800
done

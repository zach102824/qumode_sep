#!/bin/bash
# Wait for 3cav track to finish, then clear improve STOP and resume the n8–n16 chain.
# Detached setsid/nohup. Does not touch 3cav jobs. Gray + wmaxsat only.
cd /workspace/qumode_sep
LOG=logs/improve/hold_then_resume.out
exec >>"$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] hold_then_resume started pid=$$"

wait_3cav_done() {
  while true; do
    sab_alive=0; cav_alive=0
    pgrep -f 'logs/3cav/q_sab_2cav\.sh' >/dev/null 2>&1 && sab_alive=1
    pgrep -f 'logs/3cav/q_3cav_screen\.sh' >/dev/null 2>&1 && cav_alive=1
    # also treat as busy while their python children still run
    pgrep -f 'run_relayout_kbit.py.*--sab 256' >/dev/null 2>&1 && sab_alive=1
    pgrep -f 'run_relayout_kbit.py.*--device 3cav' >/dev/null 2>&1 && cav_alive=1
    sab_done=0; cav_done=0
    [ -f logs/3cav/q_sab_2cav.done ] && sab_done=1
    [ -f logs/3cav/q_3cav_screen.done ] && cav_done=1
    # Prefer done flags; if scripts exited without done, fall back to not-alive
    if { [ "$sab_done" = 1 ] || [ "$sab_alive" = 0 ]; } && { [ "$cav_done" = 1 ] || [ "$cav_alive" = 0 ]; }; then
      # final settle: no 3cav python left
      if ! pgrep -f 'run_relayout_kbit.py.*--device 3cav' >/dev/null 2>&1 \
         && ! pgrep -af 'run_relayout_kbit.py' 2>/dev/null | grep -E 'sab (128|256|512)|q_sab_2cav|3cav' >/dev/null; then
        # still may have leftover sab256 workers under the bash parent — require bash gone or done
        if [ "$sab_alive" = 0 ] && [ "$cav_alive" = 0 ]; then
          echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] 3cav track idle (sab_done=$sab_done cav_done=$cav_done)"
          return 0
        fi
      fi
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] waiting 3cav sab_alive=$sab_alive cav_alive=$cav_alive sab_done=$sab_done cav_done=$cav_done"
    sleep 90
  done
}

wait_3cav_done

# Extra settle so cores are free
sleep 15
if pgrep -f 'run_relayout_kbit.py' >/dev/null 2>&1; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] leftover run_relayout still present; waiting until clear"
  while pgrep -f 'run_relayout_kbit.py' >/dev/null 2>&1; do sleep 60; done
fi

if [ -f logs/improve/STOP ]; then
  rm -f logs/improve/STOP
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] cleared logs/improve/STOP"
fi

# Resume queue51 n18/n20 if not already done
if [ ! -f logs/improve/queue51_n16cheap_n18n20.done ]; then
  if ! pgrep -f 'queue51_n16cheap_n18n20_resume\.sh' >/dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] launching resume"
    setsid nohup bash -c 'logs/improve/queue51_n16cheap_n18n20_resume.sh >> logs/improve/queue51_n16cheap_n18n20_resume.out 2>&1' </dev/null >/dev/null 2>&1 &
    echo "resume pid $!"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] resume already running"
  fi
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] queue51_n16cheap already done"
fi

# Relaunch waiters 52–63 if missing
for i in 52 53 54 55 56 57 58 59 60 61 62 63; do
  if [ -f "logs/improve/queue${i}.done" ]; then
    continue
  fi
  if pgrep -f "logs/improve/queue${i}\\.sh" >/dev/null 2>&1; then
    continue
  fi
  if [ -x "logs/improve/queue${i}.sh" ]; then
    # truncate stale empty out so we can see fresh START lines
    : > "logs/improve/queue${i}.out"
    setsid nohup bash -c "logs/improve/queue${i}.sh >> logs/improve/queue${i}.out 2>&1" </dev/null >/dev/null 2>&1 &
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] relaunched queue${i} pid $!"
  fi
done

# Restart watchdog if absent
if ! pgrep -f 'logs/improve/watchdog\.sh' >/dev/null 2>&1; then
  setsid nohup bash -c 'logs/improve/watchdog.sh >> logs/improve/watchdog.log 2>&1' </dev/null >/dev/null 2>&1 &
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] restarted watchdog pid $!"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] hold_then_resume armed complete; exiting waiter"
touch logs/improve/hold_then_resume.done

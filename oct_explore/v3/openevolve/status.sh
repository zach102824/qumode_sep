#!/bin/bash
cd /workspace/qumode_sep/oct_explore/v3/openevolve
date +%T; free -m|sed -n 2p; echo oom=$(dmesg|grep -c oom-kill)
pgrep -af "openevolve.cli|check_watch|restart_at"|awk '{print $1,$5,$6}'|sort -u -k2
cat logs/restart_at_ckpt.log
ls runs/net16o/checkpoints runs/net32o/checkpoints 2>/dev/null|tr '\n' ' '; echo
python3.13 -c "
import json,glob
for run in ['net16o','net32o']:
  for p in sorted(glob.glob(f'runs/{run}/check/checkpoint_*.json')):
    try: d=json.load(open(p))
    except Exception: print(run,p[-20:],'pending'); continue
    print(run,p.split('/')[-1],'single',round(d.get('n20_single_median') or 0,1),round(d.get('n20_single_B90') or 0,1),'rr',round(d.get('n20_rr_median') or 0,1),round(d.get('n20_rr_B90') or 0,1),d.get('error','') or '')
"

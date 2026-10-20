#!/bin/bash
cd /workspace/qumode_sep/oct_explore/v3/openevolve
echo "$(date +%T) $(free -m|awk '/Mem:/{print "avail",$7}') oom=$(dmesg|grep -c oom-kill)"
for r in net16o net32o; do pgrep -f "openevolve.cli runs/$r" >/dev/null && echo "$r alive $(ps -o stat= -p $(pgrep -d, -f "openevolve.cli runs/$r")|tr '\n' ' ')" || echo "$r DEAD"; done
pgrep -f watch32_chunked >/dev/null || echo "watch32 DEAD"; pgrep -f "check_watch.sh runs/net16o" >/dev/null || echo "check16 DEAD"; pgrep -f cursor_adapter >/dev/null || echo "adapter DEAD"
python3.13 -c "
import json,glob
for p in sorted(glob.glob('runs/net16o/check/checkpoint_*.json'))+sorted(glob.glob('runs/net32o/check/*_chunked.json')):
  try: d=json.load(open(p))
  except Exception: continue
  if d.get('n20_rr_median'): print(p.split('/')[1],p.split('/')[-1][:13],'rr',round(d['n20_rr_median'],1),round(d['n20_rr_B90'],1),'single',round(d['n20_single_median'],1),round(d['n20_single_B90'],1))
" | tail -6

with open("/workspace/qumode_sep/oct_explore/oct23_exp/run_wave3_queue.py", "r") as f:
    c = f.read()
c = c.replace('subprocess.check_output("ps aux | grep \'part2_multicav.py\' | grep -v grep | wc -l", shell=True)',
              'subprocess.check_output("ps aux | grep \'python3 /workspace/qumode_sep/oct_explore/oct23_exp/part2_multicav.py\' | grep -v grep | grep -v \'sh -c\' | wc -l", shell=True)')
with open("/workspace/qumode_sep/oct_explore/oct23_exp/run_wave3_queue.py", "w") as f:
    f.write(c)

# net16x is TOP PRIORITY on this box (Zach, 2026-10-22)
net16r is PAUSED (runs/net16r/net16r.STOP); net16x gets all compute (2 evaluation slots, no RAM floor).
Other agents/tasks must NOT stop, kill, pause, renice or starve net16x (net16x_driver.sh, net16x_watchdog.sh, net16x_checker.sh,
eval_governor16x.py, cursor_adapter16x.py on :8766, openevolve.cli runs/net16x, evaluator_net16x / chunked_check16x children).
Do not start heavy jobs that push MemAvailable below ~4 GB. Only Zach stops it (create runs/net16x/net16x.STOP).

# net16r is TOP PRIORITY on this box (Zach, 2026-10-21)
Other agents/tasks must NOT stop, kill, pause, renice or starve net16r (driver, watchdog, checker, eval_governor16r, cursor_adapter,
openevolve.cli runs/net16r, evaluator_net16r children). Keep >= 3.2 GB free RAM; do not start heavy jobs that push MemAvailable below ~6 GB.
Only Zach stops it (create runs/net16r/net16r.STOP).

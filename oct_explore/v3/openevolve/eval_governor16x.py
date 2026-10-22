"""net16x governor: at most SLOTS (default 2) running 'evaluator_net16x.py --inner' children; RAM floor.
A new child is SIGSTOPped within ~0.3 s of spawn (before training starts) unless a slot is free AND MemAvailable >= FLOOR+NEED MB.
Stopped children are resumed oldest-first when both hold. If MemAvailable drops below FLOOR while >1 child runs, the youngest is paused.
(Child wall timeouts in evaluator_net16x count only non-stopped time.)"""
import os, signal, time, sys
SLOTS = int(os.environ.get("GOV_SLOTS", "2")); FLOOR = int(os.environ.get("GOV_FLOOR_MB", "0")); NEED = int(os.environ.get("GOV_NEED_MB", "0"))
def kids():
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit(): continue
        try:
            c = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            if any(x.endswith(b"evaluator_net16x.py") for x in c) and b"--inner" in c:
                st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split(); out.append((int(st[19]), int(pid), st[0]))
        except Exception: pass
    return sorted(out)
def avail(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) // 1024
log = lambda s: print(time.strftime("%Y-%m-%d %H:%M:%S"), s, flush=True)
last = 0; last_cont = 0.0
while True:   # non-blocking loop (never sleeps long): over-slot children are paused youngest-first within ~0.3 s
    K = kids(); run = [p for _, p, s in K if s != "T"]; stp = [p for _, p, s in K if s == "T"]; a = avail()
    while len(run) > SLOTS or (len(run) > 1 and a < FLOOR):
        p = run.pop()   # youngest
        try: os.kill(p, signal.SIGSTOP); stp.append(p); log(f"STOP {p} (running={len(run)}, slots={SLOTS}, avail={a}MB)")
        except Exception: pass
    if stp and len(run) < SLOTS and a >= FLOOR + NEED and time.time() - last_cont > 60:
        p = sorted(stp)[0] if False else [q for _, q, s in K if q in stp][0]   # oldest stopped
        try: os.kill(p, signal.SIGCONT); last_cont = time.time(); log(f"CONT {p} (avail={a}MB)")
        except Exception: pass
    if time.time() - last > 600: last = time.time(); log(f"heartbeat running={len(run)} stopped={len(stp)} avail={a}MB")
    time.sleep(0.3)

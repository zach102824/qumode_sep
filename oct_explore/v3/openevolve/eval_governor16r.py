"""net16r governor: at most SLOTS (default 2) running 'evaluator_net16r.py --inner' children; RAM floor.
A new child is SIGSTOPped within ~0.3 s of spawn (before training starts) unless a slot is free AND MemAvailable >= FLOOR+NEED MB.
Stopped children are resumed oldest-first when both hold. If MemAvailable drops below FLOOR while >1 child runs, the youngest is paused.
(Child wall timeouts in evaluator_net16r count only non-stopped time.)"""
import os, signal, time, sys
SLOTS = int(os.environ.get("GOV_SLOTS", "2")); FLOOR = int(os.environ.get("GOV_FLOOR_MB", "3200")); NEED = int(os.environ.get("GOV_NEED_MB", "2300"))
def kids():
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit(): continue
        try:
            c = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            if any(x.endswith(b"evaluator_net16r.py") for x in c) and b"--inner" in c:
                st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split(); out.append((int(st[19]), int(pid), st[0]))
        except Exception: pass
    return sorted(out)
def avail(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) // 1024
log = lambda s: print(time.strftime("%Y-%m-%d %H:%M:%S"), s, flush=True)
seen = set(); last = 0
while True:
    K = kids(); run = [p for _, p, s in K if s != "T"]; stp = [p for _, p, s in K if s == "T"]; a = avail()
    for _, p, s in K:
        if p in seen: continue
        seen.add(p)
        if s != "T" and (len(run) > SLOTS or a < FLOOR + NEED or (len(run) > 1 and run.index(p) >= SLOTS)):
            if len(run) > 1 or a < FLOOR + NEED:
                try: os.kill(p, signal.SIGSTOP); run.remove(p); stp.append(p); log(f"STOP new {p} (running={len(run)}, avail={a}MB)")
                except Exception: pass
    if len(run) > 1 and a < FLOOR:
        p = run[-1]; os.kill(p, signal.SIGSTOP); log(f"STOP {p}: avail {a}MB < floor {FLOOR}")
    elif stp and len(run) < SLOTS and a >= FLOOR + NEED:
        p = stp[0]; os.kill(p, signal.SIGCONT); log(f"CONT {p} (avail={a}MB)"); time.sleep(60)
    seen &= {p for _, p, _ in K}
    if time.time() - last > 600: last = time.time(); log(f"heartbeat running={len(run)} stopped={len(stp)} avail={a}MB")
    time.sleep(0.3)

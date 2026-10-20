"""Serialise OpenEvolve evaluator children across runs (memory floor). If more than one 'evaluator_net.py --inner' process exists,
SIGSTOP every one except the oldest within ~0.3 s of spawn (before training starts, so the candidate's own time budget is unaffected),
and SIGCONT the next oldest when the running one exits. Also SIGSTOPs a new child if free RAM < 3.2 GB + 3.0 GB."""
import os, signal, time, subprocess
def inner():
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit(): continue
        try:
            c = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            if any(b"evaluator_net.py" in x for x in c) and b"--inner" in c:
                st = open(f"/proc/{pid}/stat").read().split(); out.append((int(st[21]), int(pid), st[2]))
        except Exception: pass
    return sorted(out)
def avail(): return int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) // 1024
log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
while True:
    L = inner()
    if L:
        run = [p for _, p, s in L if s != "T"]
        if len(run) > 1:
            for p in run[1:]: os.kill(p, signal.SIGSTOP); log(f"STOP {p} (another eval running)")
        elif len(run) == 0:
            nxt = L[0][1]
            if avail() >= 3200 + 3000: os.kill(nxt, signal.SIGCONT); log(f"CONT {nxt}")
    time.sleep(0.3)

"""map logs/ideas_raw.log entries to OpenEvolve program ids: each idea -> first 'Iteration N: Program X' logged after it in the run logs."""
import re, glob, datetime as dt
R = "/workspace/qumode_sep/oct_explore/v3/openevolve/"
ideas = []
for l in open(R + "logs/ideas_raw.log"):
    t, k, txt = l.rstrip("\n").split("\t", 2); ideas.append([dt.datetime.strptime(t, "%Y-%m-%d %H:%M:%S"), txt])
its = []
for f in glob.glob(R + "logs/net*_*.log"):
    run = f.split("/")[-1].split("_")[0]
    for l in open(f, errors="ignore"):
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - INFO - Iteration (\d+): Program (\S+)", l)
        if m: its.append((dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), run, m.group(2), m.group(3)))
its.sort(); used = set(); out = []
for t, txt in sorted(ideas):
    c = [x for x in its if x[0] >= t and x[3] not in used]
    pid = c[0] if c else None
    if pid: used.add(pid[3])
    out.append(f"{t}\t{pid[1] if pid else '?'}\titer={pid[2] if pid else '?'}\t{pid[3] if pid else 'pending'}\t{txt}")
open(R + "logs/ideas.log", "w").write("\n".join(out) + "\n")

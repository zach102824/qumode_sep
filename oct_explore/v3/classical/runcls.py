import json, os, subprocess, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); BIN = os.path.join(HERE, "cls")
I = json.load(open(os.path.join(HERE, "instances.json")))
def run(ins, n, R, mode, a, b=0, st=0, seed=1):
    s = f"{n} {ins['m']} {R} {mode} {a} {b} {st} {seed + ins['seed'] % 1000003}\n" + "\n".join(" ".join(map(str, v + sg)) for v, sg in ins["cl"])
    o = subprocess.run([BIN], input=s, capture_output=True, text=True, check=True).stdout.split()
    A = np.array(o, float).reshape(-1, 2); return A[:, 0].mean(), A[:, 1].mean()
def score(x): x = np.asarray(x); return (np.log(np.median(x)) + np.log(np.quantile(x, 0.9))) / 2
def stats(x): x = np.asarray(x, float); return {"median": float(np.median(x)), "B90": float(np.quantile(x, 0.9)), "mean": float(x.mean())}

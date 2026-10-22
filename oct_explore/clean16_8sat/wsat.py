"""WalkSATlm C (qaoa_ksat/walksatlm_c/wslm.c), p=0.30 w1=6 w2=5, n-flip tries with restart, 500 runs/formula on clean16_8sat test sets (k=8,
clauses exactly as generated: repeated literals / tautologies kept, as in the paper ensemble). lookups = tries + flips."""
import sys, subprocess, json, numpy as np, common as CM, sim
n = int(sys.argv[1]); RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 500; WS = "/workspace/qumode_sep/qaoa_ksat/walksatlm_c/wslm"
ins = sim.instances(n, CM.NTEST, CM.TEB); lines = []
for i in ins:
    m, cl, E = sim.gen_instance(n, i["seed"]); assert E.min() == 0
    lines.append(f"{n} {CM.K} {m} " + " ".join(str((v + 1) * s) for vs, ss in cl for v, s in zip(vs, ss)))
A = []   # one formula at a time (the all-at-once input string OOMs at n=20); per-formula seed
for fi, l in enumerate(lines):
    out = subprocess.run([WS, "run", str(n), "1", str(22345 + 1000 * n + fi), "0.30", "6", "5"], input=(l + "\n") * RUNS, capture_output=True, text=True, check=True).stdout
    A.append(np.loadtxt(out.splitlines(), ndmin=2))
A = np.stack(A)
tries, flips = A[..., 0].mean(1), A[..., 1].mean(1); look = tries + flips
q = lambda x: [float(np.median(x)), float(np.quantile(x, .9))]
r = dict(n=n, seeds=[i["seed"] for i in ins], lookups=look.tolist(), tries=tries.tolist(), lookups_mb=q(look), tries_mb=q(tries))
json.dump(r, open(f"{CM.RES}/W_n{n}.json", "w")); print(n, "lookups", r["lookups_mb"], "tries", r["tries_mb"])

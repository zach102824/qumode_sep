"""WalkSATlm C (qaoa_ksat/walksatlm_c/wslm.c), p=0.30 w1=6 w2=5, n-flip tries with restart, 500 runs/formula on clean16 test sets.
lookups = tries (fresh starts) + flips. Per formula: mean over 500 runs; then median/B90 over 100 formulas."""
import sys, subprocess, json, numpy as np, common as CM, sim
n = int(sys.argv[1]); RUNS = 500; WS = "/workspace/qumode_sep/qaoa_ksat/walksatlm_c/wslm"
ins = sim.instances(n, CM.NTEST, CM.TEB); lines = []
for i in ins:
    m, cl, E = sim.gen_instance(n, i["seed"]); assert E.min() == 0
    lines.append(f"{n} 4 {m} " + " ".join(str((v + 1) * s) for vs, ss in zip(*zip(*cl)) for v, s in zip(vs, ss)))
inp = "".join(l + "\n" for l in lines for _ in range(RUNS))
out = subprocess.run([WS, "run", str(n), "1", str(12345 + n), "0.30", "6", "5"], input=inp, capture_output=True, text=True, check=True).stdout
A = np.loadtxt(out.splitlines(), ndmin=2).reshape(len(ins), RUNS, 3)
tries, flips = A[..., 0].mean(1), A[..., 1].mean(1); look = tries + flips
q = lambda x: [float(np.median(x)), float(np.quantile(x, .9))]
r = dict(n=n, seeds=[i["seed"] for i in ins], lookups=look.tolist(), tries=tries.tolist(), lookups_mb=q(look), tries_mb=q(tries))
json.dump(r, open(f"{CM.RES}/W_n{n}.json", "w")); print(n, "lookups", r["lookups_mb"], "tries", r["tries_mb"])

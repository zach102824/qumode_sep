"""Pick ONE SATLike cfg for all n: eval candidate uniform cfgs on held-out instances (seed base 7e9, disjoint from conv 5e9 / fresh 9e8),
200 inst x 2 runs at n=12,16,20, cap min(40*2^n,4e6). Output logs/conv/uniform_tune.jsonl."""
import json, random, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.')
from noiseless import strong_classical as sc
from noiseless.make_fresh_instances import fast_energies
CANDS = {"A": {"mode": 0, "swt": 20, "sp": 0.01, "t_bms": 3}, "B": {"mode": 0, "swt": 100, "sp": 0.0, "t_bms": 15},
         "C": {"mode": 0, "swt": 20, "sp": 0.01, "t_bms": 15}, "D": {"mode": 0, "swt": 100, "sp": 0.0, "t_bms": 3}}
def inst(n, idx):
    k = idx * 1000
    while True:
        seed = 7_000_000_000 + 10_000_000 * n + k; r = random.Random(seed); k += 1; m = int(round(9.93 * n)); cl = []
        for _ in range(m):
            v = sorted(r.sample(range(n), 4)); s = [r.choice([1, -1]) for _ in v]; w = r.randint(1, 5)
            cl.append({"vars": [i + 1 for i in v], "signs": s, "w": w})
        E = fast_energies(n, cl)
        if (E == E.min()).sum() == 1: return {"n": n, "clauses": cl, "gsv": int(np.argmin(E))}
def job(a):
    n, idx = a; d = inst(n, idx); out = []
    for c, cfg in CANDS.items():
        for j in range(2):
            h, _ = sc.runone("satlike", d, 8_000_000_000 + 10_000_000 * n + 10 * idx + j, min(40 * 2 ** n, 4_000_000), cfg); out.append((c, int(h)))
    return n, idx, out
if __name__ == "__main__":
    jobs = [(n, i) for n in (12, 16, 20) for i in range(200)]
    with Pool(8) as p, open("logs/conv/uniform_tune.jsonl", "w") as f:
        for n, i, o in p.imap_unordered(job, jobs, chunksize=2): f.write(json.dumps(dict(n=n, inst=i, res=o)) + "\n"); f.flush()
    R = [json.loads(l) for l in open("logs/conv/uniform_tune.jsonl")]
    for c in CANDS:
        b = []
        for n in (12, 16, 20):
            h = np.array([x[1] for r in R if r["n"] == n for x in r["res"] if x[0] == c], float); h[h < 0] = np.inf
            b.append(np.percentile(h, 90, method="higher")); print(c, n, "B90", b[-1], "B99", np.percentile(h, 99, method="higher"), "unsolved", int(np.isinf(h).sum()))
        print(c, CANDS[c], "avgB90", np.mean(b))

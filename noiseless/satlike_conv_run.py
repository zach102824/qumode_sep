"""Convergence study: X new unique-GS instances per n (same generator as make_fresh_instances / satlike_manyinst, new disjoint seeds), Y SATLike runs each.
Instance seed 5e9+1e7*n+k (existing use 9e8+1e6*n+k, k<2.4e6 -> <9.5e8: disjoint). Run seed 6e9+1e7*n+10*idx+j."""
import json, random, sys, os
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.')
from noiseless import strong_classical as sc
from noiseless.make_fresh_instances import fast_energies
BEST = json.load(open('logs/strong/best_cfg.json')); Y = 5
def job(a):
    n, idx = a; k = idx * 1000  # each instance index scans its own k block until first unique-GS
    while True:
        seed = 5_000_000_000 + 10_000_000 * n + k; r = random.Random(seed); k += 1; m = int(round(9.93 * n)); cl = []
        for _ in range(m):
            v = sorted(r.sample(range(n), 4)); s = [r.choice([1, -1]) for _ in v]; w = r.randint(1, 5)
            cl.append({"vars": [i + 1 for i in v], "signs": s, "w": w})
        E = fast_energies(n, cl)
        if (E == E.min()).sum() == 1: break
    g = int(np.argmin(E)); u = np.unique(E)
    d = {"n": n, "m": m, "ratio": 9.93, "seed": seed, "gs": format(g, f"0{n}b"), "E_gs": int(E[g]), "E_2nd": int(u[1]), "clauses": cl}
    open(f"Hamiltonians/wmaxsat_conv/conv_n{n:02d}_{idx}.json", "w").write(json.dumps(d))
    d2 = {"n": n, "clauses": cl, "gsv": g}; hits = []
    for j in range(Y):
        h, _ = sc.runone("satlike", d2, 6_000_000_000 + 10_000_000 * n + 10 * idx + j, min(40 * 2 ** n, 4_000_000), BEST[f"satlike|{n}"]["cfg"])
        hits.append(int(h))
    return n, idx, seed, hits
if __name__ == "__main__":
    ns = [int(x) for x in sys.argv[1].split(',')]; X = int(sys.argv[2]); out = sys.argv[3]
    done = set()
    if os.path.exists(out):
        for l in open(out): r = json.loads(l); done.add((r["n"], r["inst"]))
    jobs = [(n, i) for n in ns for i in range(X) if (n, i) not in done]
    with Pool(8) as p, open(out, 'a') as f:
        for n, i, s, h in p.imap_unordered(job, jobs, chunksize=4):
            f.write(json.dumps(dict(n=n, inst=i, seed=s, hits=h)) + "\n"); f.flush()
    print("ALLDONE", flush=True)

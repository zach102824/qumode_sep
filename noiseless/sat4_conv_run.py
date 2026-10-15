"""Unweighted random 4-SAT (all w=1) replica of the SATLike convergence study (satlike_conv_run.py / satlike_uniform_tune.py).
Same generator (4 distinct vars/clause, random signs, m=round(9.93n)) minus the weight draw. Unique-GS only (same rule).
Seeds (disjoint from all existing 9e8/5e9/6e9/7e9/8e9 bases): instances 1.1e10+1e7n+k, tune instances 1.2e10+1e7n+k,
conv runs 1.3e10+1e7n+10idx+j, tune runs 1.4e10+1e7n+10idx+j. k scans block idx*1000.. (attempts recorded -> rejection rate).
usage: sat4_conv_run.py tune  |  sat4_conv_run.py run NS X OUT"""
import json, random, sys, os
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, '.')
from noiseless import strong_classical as sc
from noiseless.make_fresh_instances import fast_energies
Y = 5; CAP = lambda n: min(40 * 2 ** n, 4_000_000)
CANDS = {"A": {"mode": 0, "swt": 20, "sp": 0.01, "t_bms": 3}, "B": {"mode": 0, "swt": 100, "sp": 0.0, "t_bms": 15},
         "C": {"mode": 0, "swt": 20, "sp": 0.01, "t_bms": 15}, "D": {"mode": 0, "swt": 100, "sp": 0.0, "t_bms": 3}}
CFG = json.load(open("logs/4sat/chosen_cfg.json"))["cfg"] if os.path.exists("logs/4sat/chosen_cfg.json") else None
def gen(n, idx, base):
    k = idx * 1000; att = 0
    while True:
        seed = base + 10_000_000 * n + k; r = random.Random(seed); k += 1; att += 1; m = int(round(9.93 * n)); cl = []
        for _ in range(m):
            v = sorted(r.sample(range(n), 4)); s = [r.choice([1, -1]) for _ in v]
            cl.append({"vars": [i + 1 for i in v], "signs": s, "w": 1})
        E = fast_energies(n, cl)
        if (E == E.min()).sum() == 1: break
    g = int(np.argmin(E)); u = np.unique(E)
    return seed, att, {"n": n, "m": m, "ratio": 9.93, "seed": seed, "weighted": False, "gs": format(g, f"0{n}b"), "E_gs": int(E[g]), "E_2nd": int(u[1]), "clauses": cl}, g
def job(a):
    n, idx = a; seed, att, d, g = gen(n, idx, 11_000_000_000)
    open(f"Hamiltonians/4sat_conv/sat4_n{n:02d}_{idx}.json", "w").write(json.dumps(d))
    d2 = {"n": n, "clauses": d["clauses"], "gsv": g}
    hits = [int(sc.runone("satlike", d2, 13_000_000_000 + 10_000_000 * n + 10 * idx + j, CAP(n), CFG)[0]) for j in range(Y)]
    return dict(n=n, inst=idx, seed=seed, attempts=att, E_gs=d["E_gs"], hits=hits)
def tjob(a):
    n, idx = a; _, _, d, g = gen(n, idx, 12_000_000_000); d2 = {"n": n, "clauses": d["clauses"], "gsv": g}; out = []
    for c, cfg in CANDS.items():
        for j in range(2): out.append((c, int(sc.runone("satlike", d2, 14_000_000_000 + 10_000_000 * n + 10 * idx + j, CAP(n), cfg)[0])))
    return dict(n=n, inst=idx, E_gs=d["E_gs"], res=out)
if __name__ == "__main__":
    if sys.argv[1] == "tune":
        jobs = [(n, i) for n in (12, 16, 20) for i in range(200)]
        with Pool(8) as p, open("logs/4sat/uniform_tune.jsonl", "w") as f:
            for r in p.imap_unordered(tjob, jobs, chunksize=2): f.write(json.dumps(r) + "\n"); f.flush()
        R = [json.loads(l) for l in open("logs/4sat/uniform_tune.jsonl")]; best = None
        for c in CANDS:
            b = []
            for n in (12, 16, 20):
                h = np.array([x[1] for r in R if r["n"] == n for x in r["res"] if x[0] == c], float); h[h < 0] = np.inf
                b.append(np.percentile(h, 90, method="higher")); print(c, n, "B90", b[-1], "B99", np.percentile(h, 99, method="higher"), "unsolved", int(np.isinf(h).sum()))
            print(c, CANDS[c], "avgB90", np.mean(b), flush=True)
            if best is None or np.mean(b) < best[1]: best = (c, np.mean(b))
        json.dump({"name": best[0], "cfg": CANDS[best[0]], "avgB90": best[1]}, open("logs/4sat/chosen_cfg.json", "w")); print("CHOSEN", best)
    else:
        ns = [int(x) for x in sys.argv[2].split(',')]; X = int(sys.argv[3]); out = sys.argv[4]; done = set()
        if os.path.exists(out):
            for l in open(out): r = json.loads(l); done.add((r["n"], r["inst"]))
        jobs = [(n, i) for n in ns for i in range(X) if (n, i) not in done]
        with Pool(8) as p, open(out, 'a') as f:
            for r in p.imap_unordered(job, jobs, chunksize=4): f.write(json.dumps(r) + "\n"); f.flush()
    print("ALLDONE", flush=True)

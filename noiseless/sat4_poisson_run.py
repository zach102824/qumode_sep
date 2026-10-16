"""Poisson-m vs fixed-m SATLike difficulty on unweighted random 4-SAT (companion to sat4_conv_run.py, same cfg/cap/5 runs).
Generator identical to sat4_conv_run.gen (4 distinct sorted vars, random signs, ratio 9.93) except as per arm:
  pu: m ~ Poisson(9.93n), unique-GS filter (our rule), success = hit the unique GS.
  ps: m ~ Poisson(9.93n), keep satisfiable (E_min=0), success = any E=0 string (paper style).
  fs: m = round(9.93n),   keep satisfiable, success = any E=0 string.
Seeds (disjoint from all existing): instances 2.1e10/2.2e10/2.3e10 (pu/ps/fs) +1e7n+k, runs 2.4e10/2.5e10/2.6e10 +1e7n+10idx+j.
usage: sat4_poisson_run.py ARM NS X OUT"""
import json, random, sys, os
import numpy as np
from multiprocessing import Pool
from numba import njit
sys.path.insert(0, '.')
from noiseless import strong_classical as sc
from noiseless.make_fresh_instances import fast_energies
Y = 5; CAP = lambda n: min(40 * 2 ** n, 4_000_000)
CFG = json.load(open("logs/4sat/chosen_cfg.json"))["cfg"]
IB = {"pu": 21_000_000_000, "ps": 22_000_000_000, "fs": 23_000_000_000}
RB = {"pu": 24_000_000_000, "ps": 25_000_000_000, "fs": 26_000_000_000}

@njit(cache=True)
def run_satlike_set(n, cv, cb, w, ptr, oc, ok, sol, seed, cap, mode, swt, sp, t_bms):
    # = strong_classical.run_satlike with `x == gsv` replaced by sol[x] (identical when sol has one entry)
    np.random.seed(seed); m = cv.shape[0]; nsat = np.zeros(m, np.int64); cnt = 0; fl = 0
    unit = np.ones(m, np.int64) if mode == 0 else w.copy()
    dw = unit.copy(); scr = np.zeros(n, np.int64); age = np.zeros(n, np.int64); unsat = np.zeros(m, np.int64)
    good = np.zeros(n, np.int64)
    x = np.random.randint(0, 1 << n); cnt += 1
    if sol[x]: return cnt, fl
    sc._init(x, n, cv, cb, nsat); step = 0
    while cnt < cap:
        step += 1
        sc._scores(x, n, cb, dw, ptr, oc, ok, nsat, scr); ng = 0
        for j in range(n):
            if scr[j] > 0: good[ng] = j; ng += 1
        best = -1
        if ng > 0:
            ns = ng if ng <= t_bms else t_bms
            for i in range(ns):
                j = good[i] if ng <= t_bms else good[np.random.randint(0, ng)]
                cnt += 1
                if sol[x ^ (1 << j)]: return cnt, fl
                if best < 0 or scr[j] > scr[best] or (scr[j] == scr[best] and age[j] < age[best]): best = j
        else:
            if np.random.random() < sp:
                for c in range(m):
                    if nsat[c] > 0 and dw[c] > unit[c]: dw[c] -= unit[c]
            else:
                for c in range(m):
                    if nsat[c] == 0 and dw[c] < swt * unit[c]: dw[c] += unit[c]
            sc._scores(x, n, cb, dw, ptr, oc, ok, nsat, scr); nu = 0
            for c in range(m):
                if nsat[c] == 0: unsat[nu] = c; nu += 1
            c = unsat[np.random.randint(0, nu)]
            for k in range(4):
                j = cv[c, k]; cnt += 1
                if sol[x ^ (1 << j)]: return cnt, fl
                if best < 0 or scr[j] > scr[best] or (scr[j] == scr[best] and age[j] < age[best]): best = j
        x = sc._flip(x, best, cb, ptr, oc, ok, nsat); fl += 1; age[best] = step
    return -1, fl

def gen(arm, n, idx):
    k = idx * 1000; att = 0
    while True:
        seed = IB[arm] + 10_000_000 * n + k; r = random.Random(seed); k += 1; att += 1
        m = int(round(9.93 * n)) if arm == "fs" else max(1, int(np.random.default_rng(seed).poisson(9.93 * n)))
        cl = []
        for _ in range(m):
            v = sorted(r.sample(range(n), 4)); s = [r.choice([1, -1]) for _ in v]
            cl.append({"vars": [i + 1 for i in v], "signs": s, "w": 1})
        E = fast_energies(n, cl)
        if arm == "pu" and (E == E.min()).sum() == 1: break
        if arm != "pu" and E.min() == 0: break
    return seed, att, m, cl, E

def job(a):
    arm, n, idx = a; seed, att, m, cl, E = gen(arm, n, idx)
    sol = (E == E.min()).astype(np.uint8); A = sc.inst_arrays({"n": n, "clauses": cl})
    hits = [int(run_satlike_set(n, *A, sol, RB[arm] + 10_000_000 * n + 10 * idx + j, CAP(n), CFG["mode"], CFG["swt"], CFG["sp"], CFG["t_bms"])[0]) for j in range(Y)]
    return dict(arm=arm, n=n, inst=idx, seed=seed, attempts=att, m=m, E_gs=int(E.min()), nsol=int(sol.sum()), hits=hits)

if __name__ == "__main__":
    arms = sys.argv[1].split(','); ns = [int(x) for x in sys.argv[2].split(',')]; X = int(sys.argv[3]); out = sys.argv[4]; done = set()
    if os.path.exists(out):
        for l in open(out): r = json.loads(l); done.add((r["arm"], r["n"], r["inst"]))
    jobs = [(a, n, i) for n in ns for a in arms for i in range(X) if (a, n, i) not in done]
    with Pool(8) as p, open(out, 'a') as f:
        for r in p.imap_unordered(job, jobs, chunksize=4): f.write(json.dumps(r) + "\n"); f.flush()
    print("ALLDONE", flush=True)

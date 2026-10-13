"""Strong classical heuristics vs hybrid lookups on the 70 strict-sweep wmaxsat instances.
Arms: tuned SA (random-restart Metropolis), weighted WalkSAT (SKC, restarts), SATLike-style dynamic clause weighting (BMS).
Lookup convention: every candidate string whose cost is scored counts as 1 lookup (repeats count); each fresh random start = 1.
  SA: 1 per proposed flip.  WalkSAT: 4 per step (all 4 vars of the picked unsat clause scored).
  SATLike: BMS step = number of goodvars sampled (min(t,|good|)); stuck step = 4 (vars of a random unsat clause).
Hit = first time the ground-state string is scored. Flips (accepted moves) are recorded too.
Tuning on held-out instances (generator seeds 100000n+70000+k, unique GS), never the 70 eval instances."""
import json, sys, random, itertools, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from numba import njit
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
OUT = REPO / "logs/strong"; NS = [8, 10, 12, 14, 16, 18, 20]
SA_B90 = dict(zip(NS, [462, 1005, 2127, 3735, 5234, 9909, 18236]))

def inst_arrays(d):
    n = d["n"]; m = len(d["clauses"])
    cv = np.zeros((m, 4), np.int64); cb = np.zeros((m, 4), np.int64); w = np.zeros(m, np.int64)
    for c, cl in enumerate(d["clauses"]):
        for k, (v, s) in enumerate(zip(cl["vars"], cl["signs"])):
            cv[c, k] = n - v; cb[c, k] = 0 if s > 0 else 1
        w[c] = cl["w"]
    occ = [[] for _ in range(n)]
    for c in range(m):
        for k in range(4): occ[cv[c, k]].append((c, k))
    ptr = np.zeros(n + 1, np.int64); oc = []; ok = []
    for j in range(n): ptr[j + 1] = ptr[j] + len(occ[j]); oc += [a for a, _ in occ[j]]; ok += [b for _, b in occ[j]]
    return cv, cb, w, ptr, np.array(oc, np.int64), np.array(ok, np.int64)

def energies(d):
    n = d["n"]; idx = np.arange(2 ** n); E = np.zeros(2 ** n, np.int64)
    for cl in d["clauses"]:
        viol = np.ones(2 ** n, bool)
        for v, s in zip(cl["vars"], cl["signs"]): viol &= ((idx >> (n - v)) & 1) == (0 if s > 0 else 1)
        E += cl["w"] * viol
    return E

def heldout(n, K=10):
    out = []; k = 0
    while len(out) < K:
        r = random.Random(100000 * n + 70000 + k); k += 1; m = int(round(9.93 * n)); cl = []
        for _ in range(m):
            v = sorted(r.sample(range(n), 4)); s = [r.choice([1, -1]) for _ in v]; w = r.randint(1, 5)
            cl.append({"vars": [i + 1 for i in v], "signs": s, "w": w})
        d = {"n": n, "clauses": cl}; E = energies(d)
        if (E == E.min()).sum() > 1: continue
        d["gsv"] = int(np.argmin(E)); out.append(d)
    return out

def evalset(n):
    out = []
    for p in sorted((REPO / "Hamiltonians/wmaxsat").glob(f"default_n{n:02d}_*.json"), key=lambda q: int(q.stem.rsplit('_', 1)[1])):
        d = json.loads(p.read_text()); E = energies(d)
        assert (E == E.min()).sum() == 1 and int(np.argmin(E)) == int(d["gs"], 2) and E.min() == d["E_gs"]
        d["gsv"] = int(d["gs"], 2); out.append(d)
    return out

@njit(cache=True)
def _init(x, n, cv, cb, nsat):
    m = cv.shape[0]
    for c in range(m):
        s = 0
        for k in range(4):
            if ((x >> cv[c, k]) & 1) != cb[c, k]: s += 1
        nsat[c] = s

@njit(cache=True)
def _flip(x, j, cb, ptr, oc, ok, nsat):
    b = (x >> j) & 1
    for q in range(ptr[j], ptr[j + 1]):
        c = oc[q]
        if b != cb[c, ok[q]]: nsat[c] -= 1
        else: nsat[c] += 1
    return x ^ (1 << j)

@njit(cache=True)
def _delta(x, j, cb, w, ptr, oc, ok, nsat):  # cost change (orig weights) of flipping j; also break
    b = (x >> j) & 1; br = 0; mk = 0
    for q in range(ptr[j], ptr[j + 1]):
        c = oc[q]
        if nsat[c] == 0: mk += w[c]
        elif nsat[c] == 1 and b != cb[c, ok[q]]: br += w[c]
    return br - mk, br

@njit(cache=True)
def run_sa(n, cv, cb, w, ptr, oc, ok, gsv, seed, cap, T0, T1, L):
    np.random.seed(seed); nsat = np.zeros(cv.shape[0], np.int64); cnt = 0; fl = 0
    while cnt < cap:
        x = np.random.randint(0, 1 << n); cnt += 1
        if x == gsv: return cnt, fl
        _init(x, n, cv, cb, nsat)
        for t in range(L):
            if cnt >= cap: break
            T = T0 * (T1 / T0) ** (t / max(1, L - 1))
            j = np.random.randint(0, n); cnt += 1
            if (x ^ (1 << j)) == gsv: return cnt, fl
            dE, br = _delta(x, j, cb, w, ptr, oc, ok, nsat)
            if dE <= 0 or np.random.random() < np.exp(-dE / T):
                x = _flip(x, j, cb, ptr, oc, ok, nsat); fl += 1
    return -1, fl

@njit(cache=True)
def run_walksat(n, cv, cb, w, ptr, oc, ok, gsv, seed, cap, p, maxf):
    np.random.seed(seed); m = cv.shape[0]; nsat = np.zeros(m, np.int64); cnt = 0; fl = 0
    unsat = np.zeros(m, np.int64); br = np.zeros(4, np.int64)
    while cnt < cap:
        x = np.random.randint(0, 1 << n); cnt += 1
        if x == gsv: return cnt, fl
        _init(x, n, cv, cb, nsat)
        for t in range(maxf):
            if cnt >= cap: break
            nu = 0
            for c in range(m):
                if nsat[c] == 0: unsat[nu] = c; nu += 1
            c = unsat[np.random.randint(0, nu)]
            for k in range(4):
                j = cv[c, k]; cnt += 1
                if (x ^ (1 << j)) == gsv: return cnt, fl
                d, br[k] = _delta(x, j, cb, w, ptr, oc, ok, nsat)
            mb = br.min()
            if mb == 0 or np.random.random() >= p:
                ks = np.where(br == mb)[0]; k = ks[np.random.randint(0, len(ks))]
            else: k = np.random.randint(0, 4)
            x = _flip(x, cv[c, k], cb, ptr, oc, ok, nsat); fl += 1
    return -1, fl

@njit(cache=True)
def _scores(x, n, cb, dw, ptr, oc, ok, nsat, sc):
    for j in range(n):
        b = (x >> j) & 1; s = 0
        for q in range(ptr[j], ptr[j + 1]):
            c = oc[q]
            if nsat[c] == 0: s += dw[c]
            elif nsat[c] == 1 and b != cb[c, ok[q]]: s -= dw[c]
        sc[j] = s

@njit(cache=True)
def run_satlike(n, cv, cb, w, ptr, oc, ok, gsv, seed, cap, mode, swt, sp, t_bms):
    np.random.seed(seed); m = cv.shape[0]; nsat = np.zeros(m, np.int64); cnt = 0; fl = 0
    unit = np.ones(m, np.int64) if mode == 0 else w.copy()
    dw = unit.copy(); sc = np.zeros(n, np.int64); age = np.zeros(n, np.int64); unsat = np.zeros(m, np.int64)
    good = np.zeros(n, np.int64)
    x = np.random.randint(0, 1 << n); cnt += 1
    if x == gsv: return cnt, fl
    _init(x, n, cv, cb, nsat); step = 0
    while cnt < cap:
        step += 1
        _scores(x, n, cb, dw, ptr, oc, ok, nsat, sc); ng = 0
        for j in range(n):
            if sc[j] > 0: good[ng] = j; ng += 1
        best = -1
        if ng > 0:
            ns = ng if ng <= t_bms else t_bms
            for i in range(ns):
                j = good[i] if ng <= t_bms else good[np.random.randint(0, ng)]
                cnt += 1
                if (x ^ (1 << j)) == gsv: return cnt, fl
                if best < 0 or sc[j] > sc[best] or (sc[j] == sc[best] and age[j] < age[best]): best = j
        else:
            if np.random.random() < sp:
                for c in range(m):
                    if nsat[c] > 0 and dw[c] > unit[c]: dw[c] -= unit[c]
            else:
                for c in range(m):
                    if nsat[c] == 0 and dw[c] < swt * unit[c]: dw[c] += unit[c]
            _scores(x, n, cb, dw, ptr, oc, ok, nsat, sc); nu = 0
            for c in range(m):
                if nsat[c] == 0: unsat[nu] = c; nu += 1
            c = unsat[np.random.randint(0, nu)]
            for k in range(4):
                j = cv[c, k]; cnt += 1
                if (x ^ (1 << j)) == gsv: return cnt, fl
                if best < 0 or sc[j] > sc[best] or (sc[j] == sc[best] and age[j] < age[best]): best = j
        x = _flip(x, best, cb, ptr, oc, ok, nsat); fl += 1; age[best] = step
    return -1, fl

def grid(arm, n):
    if arm == "sa": return [dict(T0=a, T1=b, L=c * n) for a in (0.5, 1.0, 2.0, 4.0) for b in (0.05, 0.2, 0.5, 1.0) for c in (1, 2, 4, 8, 16, 32, 64) if b < a]
    if arm == "walksat": return [dict(p=a, maxf=c) for a in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5) for c in (n, 4 * n, 16 * n, 64 * n, 10 ** 9)]
    return [dict(mode=a, swt=b, sp=c, t_bms=t) for a in (0, 1) for b in (2, 5, 20, 100, 1000) for c in (0.0, 0.01, 0.1) for t in (3, 15)]

def runone(arm, d, seed, cap, cfg):
    A = inst_arrays(d); n = d["n"]
    if arm == "sa": return run_sa(n, *A, d["gsv"], seed, cap, cfg["T0"], cfg["T1"], cfg["L"])
    if arm == "walksat": return run_walksat(n, *A, d["gsv"], seed, cap, cfg["p"], cfg["maxf"])
    return run_satlike(n, *A, d["gsv"], seed, cap, cfg["mode"], cfg["swt"], cfg["sp"], cfg["t_bms"])

_CACHE = {}
def task(a):
    phase, arm, n, ci, cfg, cap = a
    key = (phase, n)
    if key not in _CACHE: _CACHE[key] = heldout(n) if phase == "tune" else evalset(n)
    hits = []
    for i, d in enumerate(_CACHE[key]):
        for s in range(20):
            seed = (7 if phase == "tune" else 20261013) + 1000 * n + 100 * i + s + (0 if phase == "tune" else 10 ** 6 * ["sa", "walksat", "satlike"].index(arm))
            h, f = runone(arm, d, seed, cap, cfg); hits.append((i, s, int(h), int(f)))
    return dict(phase=phase, arm=arm, n=n, ci=ci, cfg=cfg, cap=cap, hits=hits)

def b90(hits, cap):
    h = sorted(x[2] if x[2] > 0 else 10 ** 18 for x in hits); v = h[int(np.ceil(0.9 * len(h))) - 1]
    return v if v <= cap else None

if __name__ == "__main__":
    tf = OUT / "tune.jsonl"; ef = OUT / "eval.jsonl"
    if not (OUT / "tune.done").exists():
        jobs = [("tune", arm, n, ci, cfg, 4 * SA_B90[n]) for arm in ("sa", "walksat", "satlike") for n in NS for ci, cfg in enumerate(grid(arm, n))]
        jobs.sort(key=lambda j: -j[2])
        with Pool(8) as pool, tf.open("w") as fh:
            for r in pool.imap_unordered(task, jobs):
                fh.write(json.dumps(r) + "\n"); fh.flush()
        (OUT / "tune.done").write_text("ok\n")
    best = {}
    for l in tf.read_text().splitlines():
        r = json.loads(l); b = b90(r["hits"], r["cap"]); su = np.mean([0 < x[2] <= r["cap"] for x in r["hits"]])
        mean = np.mean([x[2] if x[2] > 0 else r["cap"] for x in r["hits"]])
        k = (b if b is not None else 10 ** 18, -su, mean); key = (r["arm"], r["n"])
        if key not in best or k < best[key][0]: best[key] = (k, r["cfg"])
    json.dump({f"{a}|{n}": dict(cfg=v[1], tune_b90=v[0][0], tune_succ=-v[0][1]) for (a, n), v in best.items()}, open(OUT / "best_cfg.json", "w"), indent=1)
    jobs = [("eval", a, n, 0, best[(a, n)][1], min(40 * 2 ** n, 4_000_000)) for a in ("sa", "walksat", "satlike") for n in NS]
    jobs.sort(key=lambda j: -j[2])
    with Pool(8) as pool, ef.open("w") as fh:
        for r in pool.imap_unordered(task, jobs):
            fh.write(json.dumps(r) + "\n"); fh.flush()
    (OUT / "eval.done").write_text("ok\n")

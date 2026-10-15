"""WalkSATlm vs WalkSAT(SKC) vs SATLike on the same 70 wmaxsat strict-sweep instances.

WalkSATlm (Cai, Luo, Su AAAI 2013): WalkSAT SKC + lmake tie-break
  lmake(x) = w1 * make1(x) + w2 * make2(x)
make1 = weighted 0-true->1-true, make2 = weighted 1-true->2-true.

Lookup convention = strong_classical.py (strict). Hit = first GS string scored.
Tunes only WalkSATlm; WalkSAT/SATLike reuse logs/strong/best_cfg.json.
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from numba import njit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "noiseless"))
OUT = REPO / "logs/walksatlm"
NS = [8, 10, 12, 14, 16, 18, 20]
SA_B90 = dict(zip(NS, [462, 1005, 2127, 3735, 5234, 9909, 18236]))

from strong_classical import (
    inst_arrays, heldout, evalset, _init, _flip, run_walksat, run_satlike,
)

@njit(cache=True)
def _break_make(x, j, cb, w, ptr, oc, ok, nsat):
    b = (x >> j) & 1
    br = 0; m1 = 0; m2 = 0
    for q in range(ptr[j], ptr[j + 1]):
        c = oc[q]
        lit_true = b != cb[c, ok[q]]
        if nsat[c] == 0:
            m1 += w[c]
        elif nsat[c] == 1:
            if lit_true:
                br += w[c]
            else:
                m2 += w[c]
    return br, m1, m2

@njit(cache=True)
def run_walksatlm(n, cv, cb, w, ptr, oc, ok, gsv, seed, cap, p, maxf, w1, w2):
    np.random.seed(seed)
    m = cv.shape[0]
    nsat = np.zeros(m, np.int64)
    cnt = 0; fl = 0
    unsat = np.zeros(m, np.int64)
    br = np.zeros(4, np.int64)
    lm = np.zeros(4, np.int64)
    while cnt < cap:
        x = np.random.randint(0, 1 << n)
        cnt += 1
        if x == gsv:
            return cnt, fl
        _init(x, n, cv, cb, nsat)
        for t in range(maxf):
            if cnt >= cap:
                break
            nu = 0
            for c in range(m):
                if nsat[c] == 0:
                    unsat[nu] = c; nu += 1
            c = unsat[np.random.randint(0, nu)]
            for k in range(4):
                j = cv[c, k]
                cnt += 1
                if (x ^ (1 << j)) == gsv:
                    return cnt, fl
                b, m1, m2 = _break_make(x, j, cb, w, ptr, oc, ok, nsat)
                br[k] = b
                lm[k] = w1 * m1 + w2 * m2
            mb = br.min()
            if mb == 0 or np.random.random() >= p:
                best_lm = -1; nbest = 0
                cand = np.zeros(4, np.int64)
                for k in range(4):
                    if br[k] == mb:
                        if lm[k] > best_lm:
                            best_lm = lm[k]; nbest = 1; cand[0] = k
                        elif lm[k] == best_lm:
                            cand[nbest] = k; nbest += 1
                k = cand[np.random.randint(0, nbest)]
            else:
                k = np.random.randint(0, 4)
            x = _flip(x, cv[c, k], cb, ptr, oc, ok, nsat)
            fl += 1
    return -1, fl


def lm_grid(n):
    # 4-SAT Cai defaults (3,1); Boulebnane used (6,5) p=0.15; extras for weighted MaxSAT
    return [dict(p=a, maxf=c, w1=w1, w2=w2)
            for a in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
            for c in (n, 4 * n, 16 * n, 64 * n, 10 ** 9)
            for (w1, w2) in ((3, 1), (6, 5), (5, 4), (4, 2), (1, 0))]


def runone(arm, d, seed, cap, cfg):
    A = inst_arrays(d); n = d["n"]
    if arm == "walksat":
        return run_walksat(n, *A, d["gsv"], seed, cap, cfg["p"], cfg["maxf"])
    if arm == "walksatlm":
        return run_walksatlm(n, *A, d["gsv"], seed, cap, cfg["p"], cfg["maxf"], cfg["w1"], cfg["w2"])
    return run_satlike(n, *A, d["gsv"], seed, cap, cfg["mode"], cfg["swt"], cfg["sp"], cfg["t_bms"])


_CACHE = {}

def task(a):
    phase, arm, n, ci, cfg, cap = a
    key = (phase, n)
    if key not in _CACHE:
        _CACHE[key] = heldout(n) if phase == "tune" else evalset(n)
    hits = []
    arm_idx = {"walksat": 0, "walksatlm": 1, "satlike": 2}[arm]
    for i, d in enumerate(_CACHE[key]):
        for s in range(20):
            seed = (7 if phase == "tune" else 20261015) + 1000 * n + 100 * i + s + (
                0 if phase == "tune" else 10 ** 6 * arm_idx)
            h, f = runone(arm, d, seed, cap, cfg)
            hits.append((i, s, int(h), int(f)))
    return dict(phase=phase, arm=arm, n=n, ci=ci, cfg=cfg, cap=cap, hits=hits)


def b90(hits, cap):
    h = sorted(x[2] if x[2] > 0 else 10 ** 18 for x in hits)
    v = h[int(np.ceil(0.9 * len(h))) - 1]
    return v if v <= cap else None


def summarize(hits, cap):
    vals = [x[2] if x[2] > 0 else None for x in hits]
    solved = [v for v in vals if v is not None and v <= cap]
    succ = len(solved) / max(1, len(vals))
    allv = np.array([v if (v is not None and v <= cap) else 1e18 for v in vals], float)
    med = float(np.median(allv))
    if med >= 1e17:
        med = None
    flips = [x[3] for x in hits if 0 < x[2] <= cap]
    return dict(
        succ=succ, median=med, b90=b90(hits, cap),
        mean=float(np.mean([v if (v is not None and v <= cap) else cap for v in vals])),
        unsolved=len(vals) - len(solved),
        b90_flips=float(np.percentile(flips, 90)) if flips else None,
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ns = [int(x) for x in os.environ.get("WL_NS", "8,10,12,14,16,18,20").split(",")]
    strong = json.load(open(REPO / "logs/strong/best_cfg.json"))

    # --- tune WalkSATlm only ---
    tf = OUT / "tune.jsonl"
    if not (OUT / "tune.done").exists():
        jobs = [("tune", "walksatlm", n, ci, cfg, 4 * SA_B90[n])
                for n in ns for ci, cfg in enumerate(lm_grid(n))]
        jobs.sort(key=lambda j: -j[2])
        print(f"tuning WalkSATlm: {len(jobs)} configs, 8 cores", flush=True)
        with Pool(8) as pool, tf.open("w") as fh:
            done = 0
            for r in pool.imap_unordered(task, jobs):
                fh.write(json.dumps(r) + "\n"); fh.flush()
                done += 1
                if done % 50 == 0:
                    print(f"  tune {done}/{len(jobs)}", flush=True)
        (OUT / "tune.done").write_text("ok\n")

    best_lm = {}
    for l in tf.read_text().splitlines():
        r = json.loads(l)
        b = b90(r["hits"], r["cap"])
        su = float(np.mean([0 < x[2] <= r["cap"] for x in r["hits"]]))
        mean = float(np.mean([x[2] if x[2] > 0 else r["cap"] for x in r["hits"]]))
        k = (b if b is not None else 10 ** 18, -su, mean)
        key = r["n"]
        if key not in best_lm or k < best_lm[key][0]:
            best_lm[key] = (k, r["cfg"])
    best = {}
    for n in ns:
        best[("walksatlm", n)] = best_lm[n][1]
        best[("walksat", n)] = strong[f"walksat|{n}"]["cfg"]
        best[("satlike", n)] = strong[f"satlike|{n}"]["cfg"]
    json.dump({f"{a}|{n}": best[(a, n)] for a, n in best}, open(OUT / "best_cfg.json", "w"), indent=1)
    print("best configs:", json.dumps({f"{a}|{n}": best[(a, n)] for a, n in best}, indent=1), flush=True)

    # --- eval all three ---
    ef = OUT / "eval.jsonl"
    if not (OUT / "eval.done").exists():
        jobs = [("eval", a, n, 0, best[(a, n)], min(40 * 2 ** n, 4_000_000))
                for a in ("walksat", "walksatlm", "satlike") for n in ns]
        jobs.sort(key=lambda j: -j[2])
        print(f"eval {len(jobs)} jobs", flush=True)
        with Pool(8) as pool, ef.open("w") as fh:
            for r in pool.imap_unordered(task, jobs):
                sm = summarize(r["hits"], r["cap"])
                r["summary"] = sm
                # drop hits from printed size but keep in file for audit
                fh.write(json.dumps(r) + "\n"); fh.flush()
                print(f"  {r['arm']:10s} n={r['n']:2d} B90={sm['b90']} med={sm['median']} succ={sm['succ']:.3f} cfg={r['cfg']}", flush=True)
        (OUT / "eval.done").write_text("ok\n")

    print("\n=== FINAL ===")
    for l in ef.read_text().splitlines():
        r = json.loads(l); sm = r["summary"]
        print(f"n={r['n']:2d} {r['arm']:10s} B90={sm['b90']} med={sm['median']} succ={sm['succ']:.3f}")

"""Alternative classical stages for the hybrid (OVERNIGHT_IMPROVEMENTS). Selected by env QM_CLS_ALGO (default: unchanged SA).
Both return `seen` {state: E} of every scored string and set .last_raw = number of scored strings (repeats counted).
  satlike : SATLike-3.0-style dynamic clause weighting, rule (all n): unit weights, swt=20, sp=0.01, BMS t=3, no restarts.
            Lookup = every candidate flip scored (BMS samples, or the 4 vars of a random unsat clause) + 1 start.
  tunedsa : random-restart Metropolis, absolute T 2.0 -> 1.0 over chains of 16n proposals, no probes; 1 lookup per proposal/start.
Clauses for satlike are recovered from Hamiltonians/wmaxsat by matching the unique GS (argmin E) and E_gs."""
import json, os
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parents[1]

def _clauses_for(E, n):
    g = int(np.argmin(E)); emin = float(E[g])
    for p in (_REPO / "Hamiltonians/wmaxsat").glob(f"default_n{n:02d}_*.json"):
        d = json.loads(p.read_text())
        if int(d["gs"], 2) == g and abs(d["E_gs"] - emin) < 1e-9:
            cl = [([n - v for v in c["vars"]], [0 if s > 0 else 1 for s in c["signs"]]) for c in d["clauses"]]
            return cl
    raise ValueError("instance not found")

def satlike_sample(E, n, budget, rng, swt=20, sp=0.01, t=3):
    cl = _clauses_for(E, n); m = len(cl); seen = {}; cnt = [0]
    occ = [[] for _ in range(n)]
    for c, (vs, bs) in enumerate(cl):
        for j, b in zip(vs, bs): occ[j].append((c, b))
    def look(v):
        cnt[0] += 1
        if v not in seen: seen[v] = float(E[v])
    npr = int(os.environ.get("QM_SL_PROBES", "0"))  # optional counted random probes (diverse seeds), inside budget
    for v in rng.integers(0, 1 << n, npr): look(int(v))
    x = int(rng.integers(1 << n)); look(x)
    nsat = [sum(((x >> j) & 1) != b for j, b in zip(*cl[c])) for c in range(m)]
    dw = [1] * m; age = [0] * n; step = 0
    def score(j):
        b = (x >> j) & 1; s = 0
        for c, vb in occ[j]:
            if nsat[c] == 0: s += dw[c]
            elif nsat[c] == 1 and b != vb: s -= dw[c]
        return s
    rlen = int(float(os.environ.get("QM_SL_RESTART", "0")) * n); last = 0  # optional: fresh random start every rlen lookups
    while cnt[0] < budget:
        step += 1
        if rlen > 0 and cnt[0] - last >= rlen:
            last = cnt[0]; x = int(rng.integers(1 << n)); look(x); dw = [1] * m
            nsat = [sum(((x >> j) & 1) != b for j, b in zip(*cl[c])) for c in range(m)]; continue
        sc = [score(j) for j in range(n)]; good = [j for j in range(n) if sc[j] > 0]
        if good:
            cand = good if len(good) <= t else [good[int(rng.integers(len(good)))] for _ in range(t)]
        else:
            if rng.random() < sp:
                for c in range(m):
                    if nsat[c] > 0 and dw[c] > 1: dw[c] -= 1
            else:
                for c in range(m):
                    if nsat[c] == 0 and dw[c] < swt: dw[c] += 1
            sc = [score(j) for j in range(n)]
            un = [c for c in range(m) if nsat[c] == 0]
            if not un:  # cost 0 = optimum reached; spend the rest of the fixed budget on a fresh random start
                x = int(rng.integers(1 << n)); look(x)
                nsat = [sum(((x >> j) & 1) != b for j, b in zip(*cl[c])) for c in range(m)]; continue
            cand = list(cl[un[int(rng.integers(len(un)))]][0])
        best = None
        for j in cand:
            if cnt[0] >= budget: break
            look(x ^ (1 << j))
            if best is None or sc[j] > sc[best] or (sc[j] == sc[best] and age[j] < age[best]): best = j
        if best is None: break
        b = (x >> best) & 1
        for c, vb in occ[best]:
            nsat[c] += -1 if b != vb else 1
        x ^= 1 << best; age[best] = step
    satlike_sample.last_raw = cnt[0]
    return seen

def tunedsa_sample(E, n, budget, rng, T0=2.0, T1=1.0, Lmul=16):
    seen = {}; cnt = [0]
    def look(v):
        cnt[0] += 1
        if v not in seen: seen[v] = float(E[v])
        return seen[v]
    L = Lmul * n
    while cnt[0] < budget:
        v = int(rng.integers(len(E))); e = look(v)
        for s in range(L):
            if cnt[0] >= budget: break
            T = T0 * (T1 / T0) ** (s / max(1, L - 1))
            w = v ^ (1 << int(rng.integers(n))); ew = look(w)
            if ew <= e or rng.random() < np.exp(-(ew - e) / T): v, e = w, ew
    tunedsa_sample.last_raw = cnt[0]
    return seen

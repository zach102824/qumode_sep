"""Variable -> cavity assignment that minimises the compiler residual (plan section 4), plus residual statistics.
Walsh coefficients of the clause cost are sparse: clause (vars v, signs s) violated = prod_i (1 + sigma_i chi_{v_i})/2,
sigma = +1 if s>0 (violated at x_v = 0) else -1 -> coefficient prod_{i in S} sigma_i / 16 on every subset S of the clause.
Retained variance(assignment) = sum over nonempty S of coef_S^2 * [cavities(S) = one cavity or two neighbours].
Optimised by simulated annealing over variable swaps; returns perm (chain position q -> variable) for compiler.permute_vars.
"""
from __future__ import annotations
import itertools, numpy as np
import compiler as CP

def walsh_terms(n, cl, w=None):
    co = {}
    for j, (v, s) in enumerate(cl):
        wj = 1.0 if w is None else float(w[j])
        for r in range(1, 5):
            for S in itertools.combinations(range(4), r):
                key = tuple(v[i] for i in S); co[key] = co.get(key, 0.0) + wj * np.prod([1 if s[i] > 0 else -1 for i in S]) / 16
    keys = list(co); return keys, np.array([co[k] for k in keys]) ** 2

def rep_ok(cavs):
    u = set(cavs); return len(u) == 1 or (len(u) == 2 and abs(min(u) - max(u)) == 1)

def anneal(n, cl, ks, iters=4000, seed=0, w=None):
    keys, w2 = walsh_terms(n, cl, w); tot = w2.sum(); rng = np.random.default_rng(seed)
    cav_of_pos = np.repeat(np.arange(len(ks)), ks); pos = np.arange(n)   # pos[v] = chain position
    byvar = [[] for _ in range(n)]
    for t, k in enumerate(keys):
        for v in k: byvar[v].append(t)
    def ok(t): return rep_ok([cav_of_pos[pos[v]] for v in keys[t]])
    cur = np.array([ok(t) for t in range(len(keys))]); kept = (w2 * cur).sum(); best = (kept, np.argsort(pos).copy()); T0 = 0.02 * tot
    for it in range(iters):
        T = T0 * (1 - it / iters) + 1e-9
        a, b = rng.choice(n, 2, replace=False)
        if cav_of_pos[pos[a]] == cav_of_pos[pos[b]]: continue
        ts = sorted(set(byvar[a]) | set(byvar[b])); pos[a], pos[b] = pos[b], pos[a]
        new = np.array([ok(t) for t in ts]); d = (w2[ts] * (new.astype(float) - cur[ts])).sum()
        if d >= 0 or rng.random() < np.exp(d / T):
            cur[ts] = new; kept += d
            if kept > best[0]: best = (kept, np.argsort(pos).copy())
        else: pos[a], pos[b] = pos[b], pos[a]
    return best[1], 1 - best[0] / tot

def default_resid(n, cl, ks, w=None):
    keys, w2 = walsh_terms(n, cl, w); cav = np.repeat(np.arange(len(ks)), ks)
    return 1 - (w2 * np.array([rep_ok([cav[v] for v in k]) for k in keys])).sum() / w2.sum()

if __name__ == "__main__":
    import sys, os, json, time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import sim
    out = {}
    for n in (8, 10, 12, 14, 15, 16, 18, 20):
        ks = CP.LAYOUTS[n]; ins = sim.instances(n, 100, 32e9); r0, r1 = [], []; t = time.time()
        for i in ins:
            m, cl, E = sim.gen_instance(n, i["seed"]); r0.append(default_resid(n, cl, ks)); perm, rr = anneal(n, cl, ks, iters=3000 if n < 16 else 5000); r1.append(rr)
            if n <= 16 and len(r1) <= 3:   # cross-check sparse residual against the FWHT projection
                Ep = CP.permute_vars(E.astype(float), n, perm); assert abs(CP.residual_frac(Ep, ks) - rr) < 1e-9, (CP.residual_frac(Ep, ks), rr)
                assert abs(CP.residual_frac(E.astype(float), ks) - r0[-1]) < 1e-9
        out[n] = dict(ks=ks, resid_default_mean=float(np.mean(r0)), resid_default_q90=float(np.quantile(r0, 0.9)), resid_mincut_mean=float(np.mean(r1)), resid_mincut_q90=float(np.quantile(r1, 0.9)))
        print(n, ks, {k: round(v, 4) for k, v in out[n].items() if k != "ks"}, f"{time.time()-t:.0f}s", flush=True)
        json.dump(out, open("results/v3/native/residual_stats.json", "w"), indent=1)

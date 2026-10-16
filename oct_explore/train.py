"""Train shared-parameter ECD+cost-phase circuit at n_train (layer growth), freeze, evaluate at test n.
usage: python train.py --tag T_raw --layout T --cost raw --Lmax 8 --test 10 12 14 20"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize
import sim

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--layout", default="T"); ap.add_argument("--cost", default="raw")
ap.add_argument("--code", default="gray"); ap.add_argument("--prep", default="uniform"); ap.add_argument("--kind", default="meanp")
ap.add_argument("--ntrain", type=int, default=100); ap.add_argument("--nt", type=int, default=8)
ap.add_argument("--Ls", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8]); ap.add_argument("--restarts", type=int, default=6)
ap.add_argument("--maxiter", type=int, default=400); ap.add_argument("--test", type=int, nargs="*", default=[10, 12, 14, 20])
ap.add_argument("--ntest", type=int, nargs="*", default=None); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--snap", type=int, default=0); ap.add_argument("--pad", type=int, default=None)
a = ap.parse_args()
OUT = f"results/{a.tag}.json"; LAY = sim.LAYOUTS_T if a.layout == "T" else sim.LAYOUTS_C
NTEST = dict(zip(a.test, a.ntest)) if a.ntest else {10: 100, 12: 100, 14: 200, 16: 100, 18: 100, 20: 100}
TRB, TEB = 31_000_000_000, 32_000_000_000
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)

def mk(n, insts, **kw):
    return sim.Circuit(n, insts, LAY[n], tbit=(a.layout == "T"), code=a.code, cost=a.cost, prep=a.prep, snap=a.snap, pad=a.pad, **kw)

if not a.eval_only: tr = mk(a.nt, sim.instances(a.nt, a.ntrain, TRB))
f = lambda x: tr.vg(x, a.kind)
def opt(x0):
    r = minimize(f, np.asarray(x0, float), jac=True, method="L-BFGS-B", options={"maxiter": a.maxiter})
    return r.x, float(r.fun)

rng = np.random.default_rng(a.seed); PN = sim.NP + a.snap
def small_layer(): return np.concatenate([[0.3 * rng.standard_normal(), 0.3 * rng.standard_normal(), rng.uniform(0, np.pi), rng.uniform(0, 2*np.pi), 0.3 * rng.standard_normal()], 0.1 * rng.standard_normal(a.snap)])
res = json.load(open(OUT)) if a.eval_only else {"args": vars(a), "train": {}}
res["train"] = {int(k): v for k, v in res["train"].items()}
prev = None
for L in ([] if a.eval_only else a.Ls):
    t0 = time.time(); cands = []
    if prev is None:
        cands = [np.concatenate([small_layer() for _ in range(L)]) for _ in range(a.restarts)]
    else:
        Lp = prev.size // PN
        for _ in range(max(2, a.restarts // 2)):
            cands.append(np.concatenate([prev] + [small_layer() * np.array([.3, .3, .3, 1, .3] + [.3] * a.snap) for _ in range(L - Lp)]))
        # interp schedule
        xp = prev.reshape(Lp, PN); t0s, t1s = np.linspace(0, 1, Lp), np.linspace(0, 1, L)
        xi = np.stack([np.interp(t1s, t0s, xp[:, j]) if Lp > 1 else np.repeat(xp[0, j], L) for j in range(PN)], 1)
        xi[:, [0, 1, 4]] *= Lp / L; cands.append(xi.reshape(-1))
        cands.append(np.concatenate([small_layer() for _ in range(L)]))
    best = None
    for x0 in cands:
        x, fv = opt(x0)
        if best is None or fv < best[1]: best = (x, fv)
    prev = best[0]; p = tr.probs(prev)
    res["train"][L] = {"x": prev.tolist(), "loss": best[1], "train": sim.summarize(p), "sec": time.time() - t0}
    log(f"L={L} train mean_p={p.mean():.4f} med1/p={np.median(1/p):.1f} ({time.time()-t0:.0f}s)")
    json.dump(res, open(OUT, "w"), indent=1)
res.setdefault("test", {}); res["test"] = {int(k): v for k, v in res["test"].items()}
import gc
for n in ([] if a.eval_only else [a.nt]) + a.test:
    N = 100 if n == a.nt else NTEST.get(n, 100)
    ins = sim.instances(n, N, TEB); res["test"][n] = {}
    chunk = 100 if n <= 12 else (25 if n <= 16 else 5)
    for L in a.Ls:
        x = np.array(res["train"][L]["x"]); p = []
        for i in range(0, N, chunk):
            c = mk(n, ins[i:i + chunk]); p += list(c.probs(x)); del c; gc.collect()
        s = sim.summarize(p, ins); s["p"] = [float(v) for v in p]
        res["test"][n][L] = s
        log(f"test n={n} L={L} mean_p={s['mean_p']:.4f} med1/p={s['median_inv_p']:.1f} B90={s['B90_inv_p']:.1f} rand_med={s['rand_median']:.1f}")
        json.dump(res, open(OUT, "w"), indent=1)
log("ALLDONE")

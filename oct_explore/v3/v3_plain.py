"""v3 PLAN A: plain shared-angle circuit (ECD -> cost.X per layer + final ECD), massive from-scratch search at n=8.

Stages (checkpointed in results/v3/<tag>.json, resumable):
 1 screen : --ninit random inits, L-BFGS-B --screen_iter its on --screen_n n=8 train instances
 2 refine : top --refine screened + growth candidates (interpolated keep-list of --grow_from tag, if given) ->
            L-BFGS-B to convergence (--maxiter) on --ntrain n=8 instances (train seeds 3.1e10)
 3 select : keep top --keep by refined train loss; pick final by mean -log p on validation n in --val (seeds 3.3e10)
 4 test   : frozen winner (and train-best) at n=8/10/12/14/20 on test seeds 3.2e10 (n=20 under a global lock)
No QAOA init. ECD fully free (no floors / fixes / penalties). Training size n=8 only.
"""
from __future__ import annotations
import argparse, json, os, sys, time, fcntl
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
import sim, v3core as V, plainmap as PM

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--code", default="gray", choices=["gray", "binary"])
ap.add_argument("--L", type=int, required=True)
ap.add_argument("--ninit", type=int, default=200); ap.add_argument("--screen_n", type=int, default=128); ap.add_argument("--screen_iter", type=int, default=300)
ap.add_argument("--refine", type=int, default=16); ap.add_argument("--keep", type=int, default=10)
ap.add_argument("--ntrain", type=int, default=1000); ap.add_argument("--maxiter", type=int, default=2000)
ap.add_argument("--grow_from", default=""); ap.add_argument("--grow_wait_h", type=float, default=8.0)
ap.add_argument("--val", type=int, nargs="*", default=[10, 12]); ap.add_argument("--val_count", type=int, default=60)
ap.add_argument("--test", type=int, nargs="*", default=[8, 10, 12, 14, 20])
ap.add_argument("--obj", default="meanlog", choices=["meanlog", "cvar", "mix"]); ap.add_argument("--alpha", type=float, default=0.2); ap.add_argument("--w", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=500)
a = ap.parse_args()
TRB, TEB, VAB = 31_000_000_000, 32_000_000_000, 33_000_000_000
os.makedirs("results/v3", exist_ok=True); OUT = f"results/v3/{a.tag}.json"
log = lambda *s: print(time.strftime("%m-%d %H:%M:%S"), *s, flush=True)
L = a.L; P = PM.nparams(L)
res = json.load(open(OUT)) if os.path.exists(OUT) else {"args": vars(a)}
def save(): json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)


_CACHE = {}
def _fns(n):
    if n not in _CACHE:
        lay = V.Layout(n, a.code)
        def probs(x, CnD, S):
            loc, mix, g = PM.split(x, L, lay.C, lay.kmax, 1)
            return lay.probs(loc, mix, lambda l: g[l] * CnD, S)
        def loss(x, CnD, S):
            nl = -jnp.log(probs(x, CnD, S) + 1e-12)
            if a.obj == "meanlog": return jnp.mean(nl)
            k = max(1, int(round(a.alpha * nl.shape[0]))); tail = jnp.mean(jnp.sort(nl)[-k:])
            return tail if a.obj == "cvar" else jnp.mean(nl) + a.w * tail
        _CACHE[n] = (lay, jax.jit(probs), jax.jit(jax.value_and_grad(loss)))
    return _CACHE[n]


def make(n, insts, grad=True):
    lay, pj, vg = _fns(n)
    ch = a.chunk if (grad and a.obj == "meanlog") else len(insts)
    def prep(sub):
        Cn = jnp.asarray(np.stack([i["E"] * n / i["m"] for i in sub]).astype(np.float32))
        return lay.to_D(Cn), jnp.asarray(lay.succ_D(sub))
    parts = [prep(insts[i:i + ch]) for i in range(0, len(insts), ch)]
    def f(x):
        x = jnp.asarray(x, jnp.float32); tot, g = 0.0, 0.0
        for c, s in parts:
            v, gg = vg(x, c, s); w = c.shape[0] / len(insts); tot += float(v) * w; g = g + np.asarray(gg, float) * w
        return tot, g
    def p(x): return np.concatenate([np.asarray(pj(jnp.asarray(x, jnp.float32), c, s)) for c, s in parts])
    return f, p


def rand_init(rng):
    rows = []
    for _ in range(L):
        rows += [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi), rng.uniform(0, 2.0)]
    rows += [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)]
    return np.array(rows)


def grow(xp):
    Lp = (xp.size - PM.NB) // PM.NP; X = xp[: Lp * PM.NP].reshape(Lp, PM.NP); tail = xp[Lp * PM.NP:]
    t0, t1 = np.linspace(0, 1, Lp), np.linspace(0, 1, L)
    Xi = np.stack([np.interp(t1, t0, X[:, j]) for j in range(PM.NP)], 1)
    return np.concatenate([Xi.reshape(-1), tail])


def run(f, x0, maxiter):
    r = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxfun": int(maxiter * 1.5)})
    return r.x, float(r.fun), int(r.nit)


rng = np.random.default_rng(a.seed + 1000 * L + (0 if a.code == "gray" else 7))
# ---- 1 screen ----
scr = res.setdefault("screen", [])
if len(scr) < a.ninit:
    f_s, _ = make(8, sim.instances(8, a.screen_n, TRB)); inits = [rand_init(rng) for _ in range(a.ninit)]
    log(f"screen L={L} code={a.code}: {a.ninit} inits x {a.screen_iter} it on {a.screen_n} inst (resume at {len(scr)})")
    t0 = time.time()
    for i in range(len(scr), a.ninit):
        x, fv, nit = run(f_s, inits[i], a.screen_iter); scr.append({"i": i, "loss": fv, "nit": nit, "x": x.tolist()})
        if i % 10 == 9 or i == a.ninit - 1:
            save(); ls = sorted(s["loss"] for s in scr); log(f"  screen {i+1}/{a.ninit} best {ls[0]:.4f} p10 {ls[min(9,len(ls)-1)]:.4f} median {np.median(ls):.4f} ({time.time()-t0:.0f}s)")
    save()

# ---- 2 refine ----
ref = res.setdefault("refine", [])
pool = [("scr%d" % s["i"], np.array(s["x"])) for s in sorted(scr, key=lambda s: s["loss"])[: a.refine]]
if a.grow_from:
    src = f"results/v3/{a.grow_from}.json"; t0 = time.time(); wf = f"logs/v3/{a.tag}.waiting"
    while not (os.path.exists(src) and "keep" in json.load(open(src))):
        if time.time() - t0 > a.grow_wait_h * 3600: log("grow source not ready; skipping growth"); break
        open(wf, "w").close(); time.sleep(120)
    if os.path.exists(wf): os.remove(wf)
    if os.path.exists(src) and "keep" in json.load(open(src)):
        for k in json.load(open(src))["keep"]: pool.append(("grow_%s" % k["id"], grow(np.array(k["x"]))))
done_ids = {r["id"] for r in ref}
if any(pid not in done_ids for pid, _ in pool):
    f_t, p_t = make(8, sim.instances(8, a.ntrain, TRB)); log(f"refine {len(pool)} cands on {a.ntrain} inst, maxiter {a.maxiter}")
    for pid, x0 in pool:
        if pid in done_ids: continue
        t0 = time.time(); l0 = f_t(x0)[0]; x, fv, nit = run(f_t, x0, a.maxiter)
        ref.append({"id": pid, "loss0": l0, "loss": fv, "nit": nit, "sec": time.time() - t0, "x": x.tolist()}); save()
        log(f"  {pid}: {l0:.4f} -> {fv:.4f} ({nit} it, {time.time()-t0:.0f}s)")

# ---- 3 select ----
if "keep" not in res or "winner" not in res:
    keep = sorted(ref, key=lambda r: r["loss"])[: a.keep]
    vals = []
    for vn in a.val:
        _, pv = make(vn, sim.instances(vn, a.val_count, VAB), grad=False)
        vals.append([float(np.mean(-np.log(pv(np.array(k["x"])) + 1e-12))) for k in keep])
    for j, k in enumerate(keep): k["val"] = [v[j] for v in vals]; k["valmean"] = float(np.mean([v[j] for v in vals]))
    res["keep"] = keep; w = min(keep, key=lambda k: k["valmean"]); res["winner"] = w["id"]; res["trainbest"] = keep[0]["id"]; save()
    for k in keep: log(f"  keep {k['id']}: train {k['loss']:.4f} val {['%.3f' % v for v in k['val']]}")
    log(f"winner {w['id']} (train-best {keep[0]['id']})")

# ---- 4 test ----
te = res.setdefault("test", {}); keep = {k["id"]: k for k in res["keep"]}
for role in ["winner", "trainbest"]:
    kid = res[role]
    if role == "trainbest" and kid == res["winner"]: continue
    x = np.array(keep[kid]["x"]); te.setdefault(role, {"id": kid})
    for n in a.test:
        if str(n) in te[role]: continue
        N = 200 if n == 14 else 100; ins = sim.instances(n, N, TEB); ch = 50 if n <= 12 else (10 if n < 20 else 2)
        lk = open("/tmp/v3_n20.lock", "w") if n >= 18 else None
        if lk: fcntl.flock(lk, fcntl.LOCK_EX)
        t0 = time.time(); p = []
        for i in range(0, N, ch):
            _, pp = make(n, ins[i:i + ch], grad=False); p += list(pp(x))
        if lk: fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
        s = V.summ(p); te[role][str(n)] = s; save()
        log(f"test {role} n={n} L={L} med1/p={s['median_inv_p']:.1f} B90={s['B90_inv_p']:.1f} ({time.time()-t0:.0f}s)")
log("ALLDONE")

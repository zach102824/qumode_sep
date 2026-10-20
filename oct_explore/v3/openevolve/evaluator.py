"""OpenEvolve evaluator: fair single-assignment metrics (median/B90 of 1/p) on n=15 (5/5/5) validation formulas.
Simulates the templated native chain ansatz with ideal one-bit-flip mixers per cavity, chain-projected (product-measure
weighted) clause phases, optional extra per-cavity SNAP phases. combined_score = -(ln med + ln B90)/2 (higher = better).
Guards: static import whitelist; per-formula design time budget; pv vs a known solution reported (pv_acc) and
penalised if it looks like the program solved the formula classically.
"""
import os, sys, time, importlib.util, json, re, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "v3", "native"))
import sim, compiler as CP
N = 15; KS = [5, 5, 5]; NVAL = int(os.environ.get("OE_NVAL", "100")); BASE = float(os.environ.get("OE_BASE", "33e9"))
TBUDGET = float(os.environ.get("OE_TBUDGET", "1.5")); PVACC_MAX = 0.85
_VAL = None
def val_set():
    global _VAL
    if _VAL is None:
        _VAL = []
        for ins in sim.instances(N, NVAL, BASE):
            m, cl, E = sim.gen_instance(N, ins["seed"]); _VAL.append((cl, E))
    return _VAL
def _mix_mats(K):
    H = np.zeros((1 << K, 1 << K))
    for g in range(1 << K):
        for j in range(K): H[g ^ (1 << j), g] = 1
    lam, Q = np.linalg.eigh(H); return lam, Q
MIX = {5: _mix_mats(5)}
def mixer(K, b): lam, Q = MIX[K]; return (Q * np.exp(-0.5j * b * lam)) @ Q.T
BITS = ((np.arange(1 << N)[:, None] >> (N - 1 - np.arange(N))) & 1).astype(np.int8)
def viol_matrix(cl):
    V = np.ones((1 << N, len(cl)), bool)
    for j, (v, s) in enumerate(cl):
        for vv, ss in zip(v, s): V[:, j] &= BITS[:, vv] == (0 if ss > 0 else 1)
    return V.astype(np.float64)
def simulate(cl, E, d):
    m = len(cl); nm = N / m; perm = np.asarray(d["perm"], int); assert sorted(perm.tolist()) == list(range(N)), "perm invalid"
    pv = np.clip(np.asarray(d.get("pv", np.full(N, .5)), float), 1e-3, 1 - 1e-3); assert pv.shape == (N,)
    g = np.asarray(d["gamma"], float); L = len(g); assert 1 <= L <= 8, "L in 1..8"
    b = np.asarray(d["beta"], float); b = np.repeat(b[:, None], 3, 1) if b.ndim == 1 else b; assert b.shape == (L, 3)
    w = np.asarray(d.get("w", np.ones((m, L))), float); assert w.shape == (m, L)
    assert np.all(np.isfinite(g)) and np.all(np.isfinite(b)) and np.all(np.isfinite(w))
    Vx = viol_matrix(cl); S = (E == 0).astype(float)
    pvc = pv[perm]  # chain order
    psi = np.full((32, 32, 32), 1 / np.sqrt(1 << N), complex)
    for l in range(L):
        Ew = nm * g[l] * (Vx @ w[:, l])                       # var-order cost
        Ec = CP.permute_vars(Ew, N, perm)                     # chain order
        P = CP.project_product(Ec, KS, pvc, "chain")
        ph = P.reshape(32, 32, 32)
        sn = d.get("snap")
        if sn is not None:
            for c in range(3):
                t = np.asarray(sn[l][c], float); assert t.shape == (32,)
                sh = [1, 1, 1]; sh[c] = 32; ph = ph + t.reshape(sh)
        psi = psi * np.exp(-1j * ph)
        for c in range(3): psi = np.moveaxis(np.tensordot(mixer(5, b[l, c]), psi, axes=([1], [c])), 0, c)
    Sc = CP.permute_vars(S, N, perm).reshape(32, 32, 32)
    return float(np.sum(np.abs(psi) ** 2 * Sc))
def static_check(src):
    bad = re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w\.]*)", src, re.M)
    bad = [x for x in bad if x.split(".")[0] not in ("numpy", "math", "itertools", "functools", "collections")]
    for tok in ("open(", "__import__", "subprocess", "eval(", "exec(", "sys.modules", "importlib", "globals(", "fast_energies"):
        if tok in src: bad.append(tok)
    return bad
def evaluate(program_path):
    t0 = time.time(); src = open(program_path).read(); bad = static_check(src)
    if bad: return {"combined_score": -20.0, "error": f"forbidden: {bad}"}
    spec = importlib.util.spec_from_file_location("prog", program_path); mod = importlib.util.module_from_spec(spec)
    try: spec.loader.exec_module(mod)
    except Exception as e: return {"combined_score": -20.0, "error": f"import: {e}"}
    ps, accs, tds = [], [], []
    for cl, E in val_set():
        try:
            td = time.time(); d = mod.design(N, list(KS), [(list(v), list(s)) for v, s in cl]); td = time.time() - td; tds.append(td)
            if td > TBUDGET: return {"combined_score": -20.0, "error": f"design too slow {td:.2f}s > {TBUDGET}s"}
            p = simulate(cl, E, d)
        except Exception as e: return {"combined_score": -20.0, "error": f"run: {type(e).__name__}: {e}"}
        ps.append(max(p, 1e-12))
        sol = np.flatnonzero(E == 0); best = max(np.mean((np.asarray(d.get("pv", np.full(N, .5))) > .5) == BITS[s].astype(bool)) for s in sol[:64]); accs.append(best)
    inv = 1 / np.array(ps); med, b90 = float(np.median(inv)), float(np.quantile(inv, 0.9)); acc = float(np.mean(accs))
    score = -(np.log(med) + np.log(b90)) / 2
    if acc > PVACC_MAX: score -= 20.0   # looks like classical solving leaked into pv
    return {"combined_score": float(score), "median_inv_p": med, "B90_inv_p": b90, "pv_acc": acc,
            "design_s_max": float(max(tds)), "eval_s": time.time() - t0, "nval": len(ps)}
if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "initial_program.py")), indent=1))

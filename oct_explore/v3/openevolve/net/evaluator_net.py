"""OpenEvolve evaluator for the NETWORK (the gate sequence/compiler/simulator/readout are frozen and hash-locked).
Candidate program must define  train(api, budget_s) -> predict,  predict(n, V, SG, m) -> dict(gamma (F,L), beta (F,L), w (F,m,L), pv (F,n)).
Score: fair round-robin over R=4 placements (min-cut + 3 seeded random): median/B90 of 1/mean_r p_r on 100 held-out n=15 formulas.
combined_score = -(ln med + ln B90)/2. pv_acc>0.85 vs a true solution => -20 penalty (classical solving).
Env: OE_N20=1 also evaluates held-out n=20 (slow; used every 10 iterations on top programs).
"""
import os, sys, re, json, time, hashlib, importlib.util, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
FROZEN = json.load(open(os.path.join(HERE, "frozen_hashes.json")))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def verify_frozen():
    bad = [f for f, h in FROZEN.items() if sha(os.path.join(ROOT, f)) != h]
    if bad: raise RuntimeError(f"FROZEN FILES MODIFIED: {bad}")
verify_frozen()
import jax, jax.numpy as jnp
import frozen_circuit as FC
R = 4; NTR = {10: int(os.environ.get("OE_NTR10", "600")), 15: int(os.environ.get("OE_NTR15", "400"))}; NVAL = int(os.environ.get("OE_NVAL", "100"))
BUDGET = float(os.environ.get("OE_BUDGET", "420")); PVACC_MAX = 0.85
CACHE = os.path.join(HERE, "..", "runs", "datacache"); os.makedirs(CACHE, exist_ok=True)
def get_set(n, count, base, R_):
    f = os.path.join(CACHE, f"n{n}_c{count}_b{int(base)}_R{R_}.npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True); D = {k: z[k] for k in z.files}; D["sol"] = list(D["sol"]); D["m"] = int(D["m"]); D["n"] = int(D["n"]); D["R"] = int(D["R"]); return D
    D = FC.make_set(n, count, base, R_); np.savez(f, **{k: (np.array(v, dtype=object) if k == "sol" else v) for k, v in D.items()}); return D
class API:
    """only handle a candidate gets. No held-out data, no solutions, no circuit internals beyond p."""
    def __init__(self, sets): self._s = sets; self.L = FC.L; self.R = R; self.n_list = sorted(sets)
    def data(self, n): D = self._s[n]; return dict(V=D["V"].copy(), SG=D["SG"].copy(), m=int(D["m"]), F=int(D["V"].shape[0]))
    def batch(self, n, fidx, F_outs=None):
        """static (non-differentiable) circuit inputs for formulas fidx; pass to p_from inside your jit."""
        D = self._s[n]; z = dict(gamma=np.zeros((len(fidx), FC.L)), beta=np.zeros((len(fidx), FC.L)), w=np.zeros((len(fidx), D["m"], FC.L)), pv=np.zeros((len(fidx), n)))
        return FC.entries(D, fidx, z)[4:]
    def p_from(self, n, outs, b):
        """differentiable: outs per formula (F,...) jnp; b from batch(). returns p (F, R) (column 0 = min-cut placement)."""
        rep = lambda x: jnp.repeat(x, R, 0); VP, BB, PERM, S, nm = b
        p = FC.circuit_p(n, rep(outs["gamma"]), rep(outs["beta"]), rep(outs["w"]), rep(outs["pv"]), VP, BB, PERM, S, nm)
        return p.reshape(-1, R)
def static_check(src):
    imps = re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w\.]*)", src, re.M)
    bad = [x for x in imps if x.split(".")[0] not in ("numpy", "jax", "math", "functools", "itertools", "collections", "time")]
    for tok in ("open(", "__", "subprocess", "eval(", "exec(", "importlib", "globals(", "frozen", "compiler", "nsim", "assign", "sim.", "os.", "sys.", "_s[", "._s", "pickle", "np.load", "savez"):
        if tok in src: bad.append(tok)
    return bad
def check_outs(o, F, n, m):
    o = {k: jnp.asarray(o[k], jnp.float32) for k in ("gamma", "beta", "w", "pv")}
    assert o["gamma"].shape == (F, FC.L) and o["beta"].shape == (F, FC.L) and o["w"].shape == (F, m, FC.L) and o["pv"].shape == (F, n), {k: v.shape for k, v in o.items()}
    for k, v in o.items(): assert bool(jnp.all(jnp.isfinite(v))), k
    assert bool(jnp.all((o["pv"] > 0) & (o["pv"] < 1))), "pv in (0,1)"
    return o
def score_set(predict, D, chunk):
    n, F, m = D["n"], D["V"].shape[0], D["m"]
    o = check_outs(predict(n, D["V"].copy(), D["SG"].copy(), m), F, n, m)
    f = jax.jit(lambda g, b_, w, pv, VP, BB, PERM, S: FC.circuit_p(n, g, b_, w, pv, VP, BB, PERM, S, n / m))
    ps = []
    for s in range(0, F, chunk):
        idx = np.arange(s, min(F, s + chunk)); args = FC.entries(D, idx, {k: np.asarray(v)[idx] for k, v in o.items()})
        ps.append(np.asarray(f(*args[:8])).reshape(-1, R))
    P = np.concatenate(ps); pv = np.asarray(o["pv"])
    acc = float(np.mean([max(np.mean((pv[i] > .5) == (s_ == 1)) for s_ in D["sol"][i]) for i in range(F)]))
    def summ(p): ip = 1 / np.maximum(p, 1e-12); return float(np.median(ip)), float(np.quantile(ip, .9))
    return dict(rr=summ(P.mean(1)), single=summ(P[:, 0]), pv_acc=acc)
def _evaluate_inner(program_path):
    t0 = time.time()
    try: verify_frozen()
    except Exception as e: return {"combined_score": -30.0, "error": str(e)}
    src = open(program_path).read(); bad = static_check(src)
    if bad: return {"combined_score": -20.0, "error": f"forbidden: {sorted(set(bad))}"}
    sets = {n: get_set(n, NTR[n], FC.TRB, R) for n in (10, 15)}
    spec = importlib.util.spec_from_file_location("cand", program_path); mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod); tt = time.time(); predict = mod.train(API(sets), BUDGET); ttrain = time.time() - tt
        if ttrain > BUDGET * 2.0: return {"combined_score": -20.0, "error": f"train over budget {ttrain:.0f}s"}
        r15 = score_set(predict, get_set(15, NVAL, FC.VAB, R), 25)
    except Exception as e: return {"combined_score": -20.0, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    try: verify_frozen()
    except Exception as e: return {"combined_score": -30.0, "error": str(e)}
    med, b90 = r15["rr"]; sc = -(np.log(med) + np.log(b90)) / 2
    if r15["pv_acc"] > PVACC_MAX: sc -= 20.0
    out = {"combined_score": float(sc), "rr_median_inv_p": med, "rr_B90_inv_p": b90, "single_median_inv_p": r15["single"][0], "single_B90_inv_p": r15["single"][1],
           "pv_acc": r15["pv_acc"], "train_s": ttrain, "eval_s": time.time() - t0}
    if os.environ.get("OE_N20") == "1":
        r20 = score_set(predict, get_set(20, NVAL, FC.VAB, R), 1)
        out.update(n20_rr_median=r20["rr"][0], n20_rr_B90=r20["rr"][1], n20_single_median=r20["single"][0], n20_single_B90=r20["single"][1], n20_pv_acc=r20["pv_acc"])
    return out
MEM_MB = int(os.environ.get("OE_MEM_MB", "3500"))
def evaluate(program_path):
    """run the candidate in a child process with an RSS watchdog (OOM safety for the shared box)."""
    import subprocess, tempfile
    out = tempfile.mktemp(suffix=".json")
    pr = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--inner", program_path, out], env=os.environ.copy())
    peak = 0; t0 = time.time()
    while pr.poll() is None:
        try: rss = int(open(f"/proc/{pr.pid}/status").read().split("VmRSS:")[1].split()[0]) // 1024
        except Exception: rss = 0
        peak = max(peak, rss)
        if rss > MEM_MB: pr.kill(); pr.wait(); return {"combined_score": -20.0, "error": f"memory {rss}MB > {MEM_MB}MB (reduce batch sizes)"}
        if time.time() - t0 > BUDGET * 2.0 + 400: pr.kill(); pr.wait(); return {"combined_score": -20.0, "error": "wall timeout"}
        time.sleep(1.0)
    try: r = json.load(open(out)); os.remove(out)
    except Exception: return {"combined_score": -20.0, "error": f"child failed rc={pr.returncode}"}
    r["peak_rss_mb"] = peak; return r
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--inner":
    json.dump(_evaluate_inner(sys.argv[2]), open(sys.argv[3], "w")); sys.exit(0)
if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "initial_net.py")), indent=1))

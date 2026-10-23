"""clean16_8sat native best: ckpt_110 program (native_prog.py), same recipe as clean16/native.py
(OE_TRAIN_N=8,12, OE_NTR=600, R=4 placements: min-cut + 3 seeded random, OE_BUDGET=180 s, 3 seeds).
Scored round-robin (1/mean_r p_r) on val n=12 (200) and test n=8,12,16,20 (100 each).

k=8 adaptations vs clean16/native.py (documented in NOTES.md):
- Formula stream: paper 8-SAT via common.py (bases 61e9/62e9/63e9); clauses padded to M(n)=ceil(rn+8√(rn)).
- Placements: exact Walsh simulated annealing over E(x) (matches assign.anneal on 4-SAT; O(iters·2^n) not O(iters·C(m,k))).
- circuit_p: sparse (clause, violating-string) lists + segment_sum (identical math to frozen_circuit.circuit_p;
  avoids the dense (2^n, B, m, K) literal tensor that is ~15 GB at n=20, m≈3500, K=8, R=4).
- nm = 1/r (= CM.NM) instead of n/m (m varies per formula in the paper ensemble).
"""
import os, sys
os.environ.update(OE_TRAIN_N="8,12", OE_NTR="600")
import common as CM, json, time, importlib.util, numpy as np
import compiler as CP, nsim as NS, frozen_circuit as FC, jax, jax.numpy as jnp

seed = int(sys.argv[1]); BUDGET = float(os.environ.get("OE_BUDGET", "180")); OUT = f"{CM.RES}/N_s{seed}.json"
SMOKE = os.environ.get("OE_SMOKE", "") == "1"
NTR = {8: (4 if SMOKE else 600), 12: (4 if SMOKE else 600)}
NVAL = 4 if SMOKE else CM.NVAL; NTEST = 4 if SMOKE else CM.NTEST
if SMOKE: BUDGET = min(BUDGET, 30.0)
R = 4; L = FC.L
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)

# ---- fast exact Walsh anneal (inclusion-exclusion over cavity 16-submasks; bit-identical to assign.anneal) ----
_BITS = {}
def _sub16(n, ks, pos):
    """(C,16) all subsets of each cavity's 4 variable-bits (bit n-1-v)."""
    C = len(ks); cav_of_pos = np.repeat(np.arange(C), ks)
    vin = [[] for _ in range(C)]
    for v in range(n): vin[cav_of_pos[pos[v]]].append(v)
    out = np.zeros((C, 16), np.int32)
    for c in range(C):
        vin[c].sort(key=lambda v: pos[v])
        for i, v in enumerate(vin[c]):
            out[c] |= ((np.arange(16) >> i) & 1).astype(np.int32) * (1 << (n - 1 - v))
    return out

def _kept_ie(w2_all, sub16):
    C = sub16.shape[0]; s = 0.0
    for a in range(C - 1):
        s += w2_all[(sub16[a][:, None] | sub16[a + 1][None, :]).ravel()].sum()
    for a in range(1, C - 1):
        s -= w2_all[sub16[a]].sum()
    return s

def anneal_fast(n, E, ks, iters=3000, seed=0):
    C = len(ks)
    if C <= 2: return np.arange(n), 0.0
    W = CP.fwht(E.astype(np.float64)); W[0] = 0.0; w2_all = W * W; tot = w2_all.sum()
    rng = np.random.default_rng(seed)
    cav_of_pos = np.repeat(np.arange(C), ks); pos = np.arange(n)
    sub = _sub16(n, ks, pos)
    kept = _kept_ie(w2_all, sub); best = (kept, np.argsort(pos).copy()); T0 = 0.02 * tot
    for it in range(iters):
        T = T0 * (1 - it / iters) + 1e-9
        a, b = rng.choice(n, 2, replace=False)
        ca, cb = cav_of_pos[pos[a]], cav_of_pos[pos[b]]
        if ca == cb: continue
        pos[a], pos[b] = pos[b], pos[a]
        # rebuild only the two affected cavities' 16-submasks
        for c in (ca, cb):
            vs = [v for v in range(n) if cav_of_pos[pos[v]] == c]
            vs.sort(key=lambda v: pos[v])
            sm = np.zeros(16, np.int32)
            for i, v in enumerate(vs):
                sm |= ((np.arange(16) >> i) & 1).astype(np.int32) * (1 << (n - 1 - v))
            sub[c] = sm
        nk = _kept_ie(w2_all, sub); d = nk - kept
        if d >= 0 or rng.random() < np.exp(d / T):
            kept = nk
            if kept > best[0]: best = (kept, np.argsort(pos).copy())
        else:
            pos[a], pos[b] = pos[b], pos[a]
            for c in (ca, cb):
                vs = [v for v in range(n) if cav_of_pos[pos[v]] == c]
                vs.sort(key=lambda v: pos[v])
                sm = np.zeros(16, np.int32)
                for i, v in enumerate(vs):
                    sm |= ((np.arange(16) >> i) & 1).astype(np.int32) * (1 << (n - 1 - v))
                sub[c] = sm
    return best[1], 1 - best[0] / tot


# ---- sparse (clause, violating-string) lists (same as gcost.py) ----
def pairs(n, cl):
    J, X = [], []
    for j, (v, s) in enumerate(cl):
        req = {}
        for vv, ss in zip(v, s):
            need = 0 if ss > 0 else 1
            if req.get(vv, need) != need: req = None; break
            req[vv] = need
        if req is None: continue
        base = sum(b << (n - 1 - vv) for vv, b in req.items()); fr = [vv for vv in range(n) if vv not in req]
        ar = np.arange(1 << len(fr), dtype=np.int64); off = np.zeros_like(ar)
        for i, vv in enumerate(fr): off |= ((ar >> i) & 1) << (n - 1 - vv)
        X.append(base + off); J.append(np.full(len(ar), j, np.int16))
    if not X: return np.zeros(0, np.int16), np.zeros(0, np.int32)
    return np.concatenate(J), np.concatenate(X).astype(np.int32)

def xo_of_perm(n, perm):
    """chain-order index -> var-order index: E_chain[x'] = E_var[xo[x']]."""
    x = np.arange(1 << n, dtype=np.int32); xo = np.zeros_like(x)
    for q, v in enumerate(perm): xo |= ((x >> (n - 1 - q)) & 1) << (n - 1 - v)
    return xo.astype(np.int32)

MPAD = lambda n: int(np.ceil(CM.R * n + 8 * np.sqrt(CM.R * n)))
CACHE = "/tmp/clean16_8sat_ncache"; os.makedirs(CACHE, exist_ok=True)
PERM_DIR = os.path.join(CM.ROOT, "results", "v3", "native", "perm_cache"); os.makedirs(PERM_DIR, exist_ok=True)


def build_one(n, ins, R_):
    """Single-formula set dict (no cross-formula XS padding). Used for n=20 chunked scoring."""
    M = MPAD(n); seed_ = int(ins["seed"]); m, cl, E = ins["m"], ins["cl"], ins["E"]
    V = np.zeros((M, CM.K), np.int32); SG = np.zeros((M, CM.K), np.int8)
    for j, (v, s_) in enumerate(cl):
        V[j, :len(v)] = v; SG[j, :len(s_)] = s_
    perms = []
    for r in range(R_):
        if r == 0:
            if len(CP.LAYOUTS[n]) <= 2: perms.append(np.arange(n))
            else:
                pf = os.path.join(PERM_DIR, f"{int(seed_)}.npy")
                if os.path.exists(pf): perms.append(np.load(pf))
                else:
                    pm, _ = anneal_fast(n, E, CP.LAYOUTS[n], iters=3000 if n < 16 else 5000, seed=0)
                    np.save(pf, pm); perms.append(pm)
        else: perms.append(np.random.default_rng(int(seed_) * 31 + r).permutation(n))
    Ss = [np.packbits(CP.permute_vars(E.astype(np.float64), n, pm) == 0) for pm in perms]
    cj, xs = pairs(n, cl)
    sol = np.flatnonzero(E == 0)[:64]
    D = dict(V=V[None], SG=SG[None], PERM=np.stack(perms)[None], S=np.stack(Ss)[None],
             E=E.astype(np.float32)[None], CJ=cj[None], XS=xs[None],
             XO=np.stack([xo_of_perm(n, pm) for pm in perms])[None],
             seed=np.array([seed_]), mr=np.array([m], np.float32),
             sol=[((sol[:, None] >> (n - 1 - np.arange(n))[None]) & 1).astype(np.int8)],
             m=M, M=M, P=len(xs), n=n, R=R_)
    return D

def get_set(n, count, base, R_):
    """Build / load a formula set with R placements. Clauses padded to M(n); sparse CJ/XS for circuit_p.
    Refuses n>=20 full-set materialisation (XS padding ~10GB); use build_one + chunked scoring instead."""
    if n >= 20 and count > 4:
        raise RuntimeError(f"get_set n={n} count={count}: use chunked build_one path (OE n>=20)")
    f = os.path.join(CACHE, f"n{n}_c{count}_b{int(base)}_R{R_}.npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True); D = {k: z[k] for k in z.files}
        D["sol"] = list(D["sol"]); D["m"] = int(D["m"]); D["n"] = int(D["n"]); D["R"] = int(D["R"]); D["M"] = int(D["M"]); D["P"] = int(D["P"])
        return D
    import sim
    M = MPAD(n); out = dict(V=[], SG=[], PERM=[], S=[], E=[], CJ=[], XS=[], XO=[], seed=[], sol=[], mr=[]); k = c = 0; t0 = time.time()
    while c < count:
        seed_ = int(base) + 10_000_000 * n + k; k += 1
        m, cl, E = sim.gen_instance(n, seed_)
        if E.min() != 0: continue
        c += 1
        V = np.zeros((M, CM.K), np.int32); SG = np.zeros((M, CM.K), np.int8)
        for j, (v, s) in enumerate(cl):
            V[j, :len(v)] = v; SG[j, :len(s)] = s
        perms = []
        for r in range(R_):
            if r == 0:
                if len(CP.LAYOUTS[n]) <= 2: perms.append(np.arange(n))
                else:
                    pf = os.path.join(PERM_DIR, f"{int(seed_)}.npy")
                    if os.path.exists(pf): perms.append(np.load(pf))
                    else:
                        pm, _ = anneal_fast(n, E, CP.LAYOUTS[n], iters=3000 if n < 16 else 5000, seed=0)
                        np.save(pf, pm); perms.append(pm)
            else: perms.append(np.random.default_rng(int(seed_) * 31 + r).permutation(n))
        Ss = [np.packbits(CP.permute_vars(E.astype(np.float64), n, pm) == 0) for pm in perms]
        cj, xs = pairs(n, cl)
        sol = np.flatnonzero(E == 0)[:64]
        out["V"].append(V); out["SG"].append(SG); out["PERM"].append(np.stack(perms)); out["S"].append(np.stack(Ss))
        out["E"].append(E.astype(np.float32)); out["CJ"].append(cj); out["XS"].append(xs)
        out["XO"].append(np.stack([xo_of_perm(n, pm) for pm in perms]))
        out["seed"].append(seed_); out["mr"].append(m)
        out["sol"].append(((sol[:, None] >> (n - 1 - np.arange(n))[None]) & 1).astype(np.int8))
        if c % 50 == 0 or c == count: log(f"  build n={n} {c}/{count} ({time.time()-t0:.0f}s)")
    P = max(len(x) for x in out["XS"])
    D = {k2: np.stack(v) for k2, v in out.items() if k2 not in ("sol", "CJ", "XS")}
    D["CJ"] = np.stack([np.pad(j, (0, P - len(j))) for j in out["CJ"]])
    D["XS"] = np.stack([np.pad(x, (0, P - len(x)), constant_values=1 << n) for x in out["XS"]])
    D["sol"] = out["sol"]; D["m"] = M; D["M"] = M; D["P"] = P; D["n"] = n; D["R"] = R_; D["mr"] = np.asarray(out["mr"], np.float32)
    np.savez(f, **{k: (np.array(v, dtype=object) if k == "sol" else v) for k, v in D.items()})
    return D

def place_of(D, fidx):
    n = D["n"]; fidx = np.asarray(fidx); PERM = D["PERM"][fidx].reshape(-1, n)
    return dict(cav=FC.cavities(n, PERM).reshape(len(fidx), R, -1, 5), ks=tuple(CP.LAYOUTS[n]))

def _corr(o, n):
    C = len(CP.LAYOUTS[n]); dS = dB = None
    if o.get("dS") is not None: dS = jnp.asarray(o["dS"], jnp.float32).reshape((-1, L, C, 32))
    if o.get("dB") is not None: dB = jnp.asarray(o["dB"], jnp.float32).reshape((-1, L, C - 1, 32, 32))
    return dS, dB

def check_outs(o, F, n, m):
    C = len(CP.LAYOUTS[n]); ex = {}
    if o.get("dS") is not None: ex["dS"] = jnp.asarray(o["dS"], jnp.float32); assert ex["dS"].shape == (F, R, L, C, 32), ex["dS"].shape
    if o.get("dB") is not None: ex["dB"] = jnp.asarray(o["dB"], jnp.float32); assert ex["dB"].shape == (F, R, L, C - 1, 32, 32), ex["dB"].shape
    o = {k: jnp.asarray(o[k], jnp.float32) for k in ("gamma", "beta", "w", "pv")}; o.update(ex)
    assert o["gamma"].shape == (F, L) and o["beta"].shape == (F, L) and o["w"].shape == (F, m, L) and o["pv"].shape == (F, n), {k: v.shape for k, v in o.items()}
    for k, v in o.items(): assert bool(jnp.all(jnp.isfinite(v))), f"output {k} has NaN/inf"
    assert bool(jnp.all((o["pv"] > 0) & (o["pv"] < 1))), "pv in (0,1)"
    return o

# ---- sparse circuit_p (identical math to FC.circuit_p; verified bit-level on n=8,12) ----
def circuit_p_sparse(n, gamma, beta, w, pv, CJ, XS, XO, PERM, S, E, dS=None, dB=None):
    """B = F*R entries. w (B,M,L); CJ/XS (B,P) var-order; XO (B,2^n) chain->var; E (B,2^n) var-order energy.
    nm = CM.NM. Returns p (B,)."""
    ks = CP.LAYOUTS[n]; B = S.shape[0]; N = 1 << n
    Wx = jax.vmap(lambda wb, cj, xs: jax.ops.segment_sum(wb[cj.astype(jnp.int32)], xs.astype(jnp.int32), N + 1)[:N])(w, CJ, XS)  # (B,N,L)
    Ew_var = CM.NM * (gamma[:, None, :] * E[:, :, None] + Wx)                                                                # (B,N,L)
    Ew = jnp.take_along_axis(Ew_var, XO[:, :, None], axis=1)                                                                  # chain order
    pvc = jnp.take_along_axis(pv, PERM.astype(jnp.int32), axis=1)
    psi = NS.uniform_psi0(ks, B)
    for l in range(L):
        Ewl = Ew[:, :, l]
        Ewl = FC.project_product_j(Ewl, pvc, n, ks)
        if dS is not None:
            I = FC.cav_idx(n)
            for c in range(len(ks)): Ewl = Ewl + jnp.take(dS[:, l, c], jnp.asarray(I[c]), axis=1)
        if dB is not None:
            I = FC.cav_idx(n)
            for a in range(len(ks) - 1): Ewl = Ewl + jnp.take(dB[:, l, a].reshape(B, -1), jnp.asarray(I[a] * 32 + I[a + 1]), axis=1)
        psi = psi * jnp.exp(-1j * Ewl.reshape(psi.shape)).astype(jnp.complex64)
        for c, k in enumerate(ks): psi = NS.apply_cav(FC.MIXK[k](beta[:, l]), psi, c, len(ks))
    return jnp.sum(jnp.abs(psi.reshape(B, -1)) ** 2 * S, 1)

def entries_sparse(D, fidx, outs):
    """Expand per-formula outs to (formula, r) entries for circuit_p_sparse."""
    n, R_ = D["n"], D["R"]; fidx = np.asarray(fidx); F = len(fidx)
    PERM = D["PERM"][fidx].reshape(F * R_, n)
    S = np.unpackbits(D["S"][fidx].reshape(F * R_, -1), axis=-1)[:, : 1 << n].astype(np.float32)
    E = np.repeat(D["E"][fidx], R_, 0)
    CJ = np.repeat(D["CJ"][fidx], R_, 0); XS = np.repeat(D["XS"][fidx], R_, 0)
    XO = D["XO"][fidx].reshape(F * R_, 1 << n)
    rep = lambda x: jnp.repeat(jnp.asarray(x), R_, 0)
    return (rep(outs["gamma"]), rep(outs["beta"]), rep(outs["w"]), rep(outs["pv"]),
            jnp.asarray(CJ), jnp.asarray(XS), jnp.asarray(XO), jnp.asarray(PERM), jnp.asarray(S), jnp.asarray(E))

# ---- API (mirrors evaluator_net.API; uses sparse circuit_p) ----
class API:
    def __init__(self, sets): self._s = sets; self.L = L; self.R = R; self.n_list = sorted(sets)
    def data(self, n): D = self._s[n]; return dict(V=D["V"].copy(), SG=D["SG"].copy(), m=int(D["m"]), F=int(D["V"].shape[0]))
    def batch(self, n, fidx):
        D = self._s[n]; fidx = np.asarray(fidx); F = len(fidx)
        # static inputs the train loop passes through to p_from (CJ/XS/XO/PERM/S/E)
        return (jnp.asarray(np.repeat(D["CJ"][fidx], R, 0)), jnp.asarray(np.repeat(D["XS"][fidx], R, 0)),
                jnp.asarray(D["XO"][fidx].reshape(F * R, 1 << n)), jnp.asarray(D["PERM"][fidx].reshape(F * R, n)),
                jnp.asarray(np.unpackbits(D["S"][fidx].reshape(F * R, -1), axis=-1)[:, : 1 << n].astype(np.float32)),
                jnp.asarray(np.repeat(D["E"][fidx], R, 0)))
    def place(self, n, fidx): return place_of(self._s[n], fidx)
    def p_from(self, n, outs, b, glob=False):
        CJ, XS, XO, PERM, S, E = b; rep = lambda x: jnp.repeat(x, R, 0)
        dS, dB = _corr(outs, n)
        p = circuit_p_sparse(n, rep(outs["gamma"]), rep(outs["beta"]), rep(outs["w"]), rep(outs["pv"]),
                             CJ, XS, XO, PERM, S, E, dS, dB)
        return p.reshape(-1, R)

# ---- load program, train, score ----
spec = importlib.util.spec_from_file_location("cand", os.path.join(CM.HERE, "native_prog.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); mod.SEED = seed
sets = {n: get_set(n, NTR[n], CM.TRB, R) for n in (8, 12)}; log("train sets", {n: s["V"].shape for n, s in sets.items()})
t = time.time(); predict = mod.train(API(sets), BUDGET); log(f"trained {time.time()-t:.0f}s")

_JIT = {}
def score(D):
    n, F, m = D["n"], D["V"].shape[0], D["m"]
    o = check_outs(predict(n, D["V"].copy(), D["SG"].copy(), m, place=place_of(D, np.arange(F))), F, n, m)
    if n not in _JIT:
        _JIT[n] = jax.jit(lambda g, b_, w, pv, CJ, XS, XO, PERM, S, E, dS, dB, n=n: circuit_p_sparse(n, g, b_, w, pv, CJ, XS, XO, PERM, S, E, dS, dB))
    f = _JIT[n]; ch = 8 if n <= 12 else (2 if n <= 16 else 1); P = []
    for s in range(0, F, ch):
        idx = np.arange(s, min(F, s + ch)); args = entries_sparse(D, idx, {k: np.asarray(o[k])[idx] for k in ("gamma", "beta", "w", "pv")})
        dS, dB = _corr({k: (o[k][idx] if k in o else None) for k in ("dS", "dB")}, n)
        if n < 20: P.append(np.asarray(f(*args, dS, dB)).reshape(-1, R))
        else:
            # n=20: one (formula, placement) entry at a time to bound memory; identical math
            P.append(np.array([[float(f(*[x[e:e + 1] for x in args], dS[e:e + 1], dB[e:e + 1])[0]) for e in range(R)]]))
    return np.concatenate(P)

RESUME = os.environ.get("OE_RESUME", "") == "1" and os.path.exists(OUT)
res = dict(seed=seed, budget=BUDGET, test={}, k8_adaptations=["sparse_circuit_p", "walsh_anneal_ie", "pad_M", "nm=1/r", "K-generalised_geom/local_tables"])
if RESUME:
    old = json.load(open(OUT)); res["test"] = old.get("test", {})
    log(f"RESUME: keeping existing tags {sorted(res['test'])}")
need = lambda tag, cnt: not (RESUME and tag in res["test"] and len(res["test"][tag].get("p", [])) >= cnt)  # partial (<cnt) => re-enter and continue
for tag, n, cnt, base in [("val", CM.VAL_N, NVAL, CM.VAB)] + [(str(n), n, NTEST, CM.TEB) for n in CM.TEST_N]:
    if not need(tag, cnt):
        log(f"skip {tag} n={n} (already have {len(res['test'][tag]['p'])} formulas)"); continue
    t = time.time()
    if n >= 20:
        # Chunked: 1 formula at a time; incremental JSON save; jax.clear_caches to bound RSS.
        import sim
        insts, k = [], 0
        while len(insts) < cnt:
            seed_ = int(base) + 10_000_000 * n + k; k += 1
            m, cl, E = sim.gen_instance(n, seed_)
            if E.min() == 0: insts.append(dict(seed=seed_, m=m, cl=cl, E=E))
        # Resume partial n=20 if present
        start_i = 0; Ps, seeds = [], []
        if tag in res["test"] and 0 < len(res["test"][tag].get("p", [])) < cnt:
            oldp = res["test"][tag]
            # Rebuild Ps from saved P rows if present, else from p as 1-col (rr only — lose per-placement)
            if "P" in oldp and len(oldp["P"]) == len(oldp["p"]):
                Ps = [np.asarray(row, float)[None, :] for row in oldp["P"]]
            else:
                Ps = [np.full((1, R), float(pi)) for pi in oldp["p"]]  # fallback: all placements = mean
            seeds = list(oldp.get("seeds", []))[:len(Ps)]
            start_i = len(Ps); log(f"  resume {tag} n={n} from {start_i}/{cnt}")
        for i0 in range(start_i, cnt):
            D1 = build_one(n, insts[i0], R); P1 = np.asarray(score(D1)); Ps.append(P1); seeds.append(int(insts[i0]["seed"]))
            if (i0 + 1) % 5 == 0:
                try: jax.clear_caches()
                except Exception: pass
            if (i0 + 1) % 10 == 0 or i0 + 1 == cnt:
                Ptmp = np.concatenate(Ps); ptmp = Ptmp.mean(1); iptmp = 1 / np.maximum(ptmp, 1e-12)
                res["test"][tag] = dict(median_inv_p=float(np.median(iptmp)), B90_inv_p=float(np.quantile(iptmp, .9)),
                                        p=ptmp.tolist(), P=Ptmp.tolist(), seeds=list(seeds))
                json.dump(res, open(OUT, "w")); log(f"  {tag} n={n} scored {i0+1}/{cnt} ({time.time()-t:.0f}s)")
        P = np.concatenate(Ps); p = P.mean(1); ip = 1 / np.maximum(p, 1e-12)
    else:
        D = get_set(n, cnt, base, R); P = score(D); p = P.mean(1); ip = 1 / np.maximum(p, 1e-12); seeds = [int(x) for x in D["seed"]]
    res["test"][tag] = dict(median_inv_p=float(np.median(ip)), B90_inv_p=float(np.quantile(ip, .9)), p=p.tolist(), P=P.tolist(), seeds=seeds)
    log(f"{tag} n={n}: rr {np.median(ip):.2f}/{np.quantile(ip,.9):.2f} ({time.time()-t:.0f}s)"); json.dump(res, open(OUT, "w"))
log("DONE")

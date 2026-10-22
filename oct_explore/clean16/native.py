"""clean16 native best: net16r ckpt_110 program (native_prog.py = best_program.py, only the training-RNG seed exposed),
trained with its own train(api, budget) on fresh n=8+12 sets exactly as the net16r evaluator does
(OE_TRAIN_N=8,12, OE_NTR=600, R=4 placements: min-cut + 3 seeded random, OE_BUDGET=180 s, 2 threads).
Scored round-robin (1/mean_r p_r) on val n=12 (200) and test n=8,12,16,20 (100 each); frozen circuit_p unchanged."""
import os, sys
os.environ.update(OE_TRAIN_N="8,12", OE_NTR="600")
import common as CM, json, time, importlib.util, numpy as np
import evaluator_net as EV, frozen_circuit as FC, jax, jax.numpy as jnp
seed = int(sys.argv[1]); BUDGET = float(os.environ.get("OE_BUDGET", "180")); OUT = f"{CM.RES}/N_s{seed}.json"
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
spec = importlib.util.spec_from_file_location("cand", os.path.join(CM.HERE, "native_prog.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); mod.SEED = seed
sets = {n: EV.get_set(n, EV.NTR[n], CM.TRB, EV.R) for n in (8, 12)}; log("train sets", {n: s["V"].shape for n, s in sets.items()})
t = time.time(); predict = mod.train(EV.API(sets), BUDGET); log(f"trained {time.time()-t:.0f}s")
def score(D):
    n, F, m = D["n"], D["V"].shape[0], D["m"]
    o = EV.check_outs(predict(n, D["V"].copy(), D["SG"].copy(), m, place=EV.place_of(D, np.arange(F))), F, n, m)
    f = jax.jit(lambda g, b_, w, pv, VP, BB, PERM, S, dS, dB: FC.circuit_p(n, g, b_, w, pv, VP, BB, PERM, S, n / m, dS, dB))
    ch = 25 if n <= 12 else (2 if n <= 16 else 1); P = []
    for s in range(0, F, ch):
        idx = np.arange(s, min(F, s + ch)); args = FC.entries(D, idx, {k: np.asarray(o[k])[idx] for k in ("gamma", "beta", "w", "pv")})
        dS, dB = EV._corr({k: (o[k][idx] if k in o else None) for k in ("dS", "dB")}, n)
        if n < 20: P.append(np.asarray(f(*args[:8], dS, dB)).reshape(-1, EV.R))
        else:   # n=20 (2^20 states): one (formula, placement) entry at a time to bound memory; identical math
            P.append(np.array([[float(f(*[x[e:e + 1] for x in args[:8]], dS[e:e + 1], dB[e:e + 1])[0]) for e in range(EV.R)]]))
    return np.concatenate(P)
res = dict(seed=seed, budget=BUDGET, test={})
for tag, n, cnt, base in [("val", CM.VAL_N, CM.NVAL, CM.VAB)] + [(str(n), n, CM.NTEST, CM.TEB) for n in CM.TEST_N]:
    t = time.time(); D = EV.get_set(n, cnt, base, EV.R); P = score(D); p = P.mean(1); ip = 1 / np.maximum(p, 1e-12)
    res["test"][tag] = dict(median_inv_p=float(np.median(ip)), B90_inv_p=float(np.quantile(ip, .9)), p=p.tolist(), P=P.tolist(), seeds=[int(x) for x in D["seed"]])
    log(f"{tag} n={n}: rr {np.median(ip):.2f}/{np.quantile(ip,.9):.2f} ({time.time()-t:.0f}s)"); json.dump(res, open(OUT, "w"))
log("DONE")

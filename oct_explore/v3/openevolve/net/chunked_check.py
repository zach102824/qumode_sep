"""Official TEST checker with lower peak memory. NOT hash-locked; it uses only frozen code (evaluator_net / frozen_circuit) unchanged.
Same procedure as evaluator_net._evaluate_inner (static check, train in budget, selection set, TEST set), except that scoring calls
FC.circuit_p on ONE (formula, placement) entry at a time and aggregates exactly: rr = 1/mean_r p_r per formula, single = placement 0,
summarised by median and 90th percentile. --verify N: also run the frozen score_set on the TEST set at size N, report the max abs diff in p."""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import evaluator_net as EV, frozen_circuit as FC, jax, jax.numpy as jnp
def p_entries(predict, D):
    n, F, m, R = D["n"], D["V"].shape[0], D["m"], D["R"]
    import inspect
    kw = dict(place=EV.place_of(D, np.arange(F))) if "place" in inspect.signature(predict).parameters else {}
    o = EV.check_outs(predict(n, D["V"].copy(), D["SG"].copy(), m, **kw), F, n, m)
    f = jax.jit(lambda g, b_, w, pv, VP, BB, PERM, S, dS, dB: FC.circuit_p(n, g, b_, w, pv, VP, BB, PERM, S, n / m, dS, dB))
    P = np.zeros((F, R))
    for i in range(F):
        args = FC.entries(D, np.array([i]), {k: np.asarray(o[k])[i:i + 1] for k in ("gamma", "beta", "w", "pv")})
        dS, dB = EV._corr({k: (o[k][i:i + 1] if k in o else None) for k in ("dS", "dB")}, n)
        for r in range(R):
            sl = lambda x: None if x is None else x[r:r + 1]
            P[i, r] = float(np.asarray(f(*[a[r:r + 1] for a in args[:8]], sl(dS), sl(dB)))[0])
    pv = np.asarray(o["pv"])
    acc = float(np.mean([max(np.mean((pv[i] > .5) == (s_ == 1)) for s_ in D["sol"][i]) for i in range(F)]))
    return P, acc
def summ(p): ip = 1 / np.maximum(p, 1e-12); return float(np.median(ip)), float(np.quantile(ip, .9))
def main(prog, verify):
    t0 = time.time(); EV.verify_frozen(); src = open(prog).read(); bad = EV.static_check(src)
    if bad: return {"combined_score": -20.0, "error": f"forbidden: {sorted(set(bad))}"}
    import importlib.util
    sets = {n: EV.get_set(n, EV.NTR[n], FC.TRB, EV.R) for n in EV.TRAIN_N}
    spec = importlib.util.spec_from_file_location("cand", prog); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    tt = time.time(); predict = mod.train(EV.API(sets), EV.BUDGET); ttrain = time.time() - tt
    Dv = EV.get_set(EV.VAL_N, EV.NVAL, FC.VAB, EV.R); Pv, accv = p_entries(predict, Dv)
    med, b90 = summ(Pv.mean(1)); sc = -(np.log(med) + np.log(b90)) / 2 - (20.0 if accv > EV.PVACC_MAX else 0.0)
    Dt = EV.get_set(EV.CHECK_N, int(os.environ.get("OE_NTEST", "100")), FC.TEB, EV.R); Pt, acct = p_entries(predict, Dt)
    EV.verify_frozen()
    out = {"combined_score": float(sc), "rr_median_inv_p": med, "rr_B90_inv_p": b90, "single_median_inv_p": summ(Pv[:, 0])[0], "single_B90_inv_p": summ(Pv[:, 0])[1],
           "pv_acc": accv, "train_s": ttrain, "check_n": EV.CHECK_N, "n20_rr_median": summ(Pt.mean(1))[0], "n20_rr_B90": summ(Pt.mean(1))[1],
           "n20_single_median": summ(Pt[:, 0])[0], "n20_single_B90": summ(Pt[:, 0])[1], "n20_pv_acc": acct, "chunked": True}
    if verify:
        r = EV.score_set(predict, Dt, verify)
        out["verify_frozen_score_set"] = {"rr": r["rr"], "single": r["single"]}
        out["verify_match_rr"] = [abs(a - b) / b for a, b in zip(r["rr"], (out["n20_rr_median"], out["n20_rr_B90"]))]
    out["eval_s"] = time.time() - t0; return out
if __name__ == "__main__":
    prog = sys.argv[1]; verify = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(json.dumps(main(prog, verify), indent=1))

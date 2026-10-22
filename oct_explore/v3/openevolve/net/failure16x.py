"""net16x STEP 1: failure analysis of net16r ckpt_10 best program on n=16 TEB (100) + VAB (60). Writes runs/net16x/failure16x.json."""
import os, sys, json, time, inspect, importlib.util, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import evaluator_net as EV, frozen_circuit as FC, pair_circuit as PC, jax, jax.numpy as jnp, sim
from frozen_circuit import CP
prog = sys.argv[1]; out = sys.argv[2]
sets = {n: EV.get_set(n, EV.NTR[n], FC.TRB, EV.R) for n in EV.TRAIN_N}
spec = importlib.util.spec_from_file_location("cand", prog); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
t = time.time(); predict = mod.train(EV.API(sets), EV.BUDGET); print("train", time.time() - t, flush=True)
n = 16; R = EV.R; rows = []
fstate = jax.jit(lambda g, b_, w, pv, VP, BB, PERM, S, dS, dB: PC.circuit_state(n, g, b_, w, pv, VP, BB, PERM, S, n / D["m"], dS, dB))
for name, base, cnt in (("TEB", FC.TEB, 100), ("VAB", FC.VAB, 60)):
    D = EV.get_set(n, cnt, base, R); F = cnt
    kw = dict(place=EV.place_of(D, np.arange(F))) if "place" in inspect.signature(predict).parameters else {}
    o = EV.check_outs(predict(n, D["V"].copy(), D["SG"].copy(), D["m"], **kw), F, n, D["m"])
    for i in range(F):
        idx = np.array([i]); args = FC.entries(D, idx, {k: np.asarray(o[k])[idx] for k in ("gamma", "beta", "w", "pv")})
        dS, dB = EV._corr({k: (o[k][idx] if k in o else None) for k in ("dS", "dB")}, n)
        outs = [np.asarray(x) for x in fstate(*args[:8], dS, dB)]   # list L of (R, 2^n)
        m_, cl, E = sim.gen_instance(n, int(D["seed"][i])); E = np.asarray(E, np.float64)
        Es = np.sort(np.unique(E)); deg = int((E == 0).sum()); gap = float(Es[1]) if len(Es) > 1 else 0.0
        V = D["V"][i]; per_r = []
        for r in range(R):
            perm = D["PERM"][i, r]; pos = np.argsort(perm); cav = pos[V] // 4; span = cav.max(1) - cav.min(1)
            Ec = CP.permute_vars(E, n, perm); P = outs[-1][r]
            pl = [float(outs[l][r][Ec == 0].sum()) for l in range(FC.L)]
            top = np.argsort(P)[::-1][:5]; sols = np.flatnonzero(Ec == 0)
            hd = [int(min(bin(int(x) ^ int(s)).count("1") for s in sols)) for x in top if Ec[x] > 0][:3]
            per_r.append(dict(p=pl[-1], p_layers=pl, cross=float((span > 0).mean()), nonnb=float((span > 1).mean()),
                              mass_E1=float(P[Ec == 1].sum()), mass_E2=float(P[Ec == 2].sum()), mass_E3p=float(P[Ec >= 3].sum()),
                              ipr_eff_states=float(1 / np.sum(P ** 2)), top_wrong_mass=float(max([P[x] for x in top if Ec[x] > 0] + [0])),
                              top_wrong_E=[float(Ec[x]) for x in top if Ec[x] > 0][:3], top_wrong_hamming=hd,
                              uniform_E1=float((Ec == 1).mean())))
        pv = np.asarray(o["pv"][i]); acc = max(np.mean((pv > .5) == (s_ == 1)) for s_ in D["sol"][i])
        conf = float(np.mean(np.abs(pv - .5) * 2))
        rows.append(dict(set=name, i=i, seed=int(D["seed"][i]), m=int(D["m"]), p_rr=float(np.mean([x["p"] for x in per_r])), deg=deg, gap=gap,
                         n_E1=int((E == 1).sum()), pv_acc=float(acc), pv_conf=conf, per_r=per_r))
    print(name, "done", time.time() - t, flush=True)
json.dump(rows, open(out, "w"), indent=0); print("saved", out)

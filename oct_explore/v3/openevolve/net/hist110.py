"""Dump full final |psi|^2 (placement r=0) on n=16 TEB for a program. Reuses failure16x logic; frozen files untouched."""
import os, sys, time, inspect, importlib.util, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import evaluator_net as EV, frozen_circuit as FC, pair_circuit as PC, jax, sim
from frozen_circuit import CP
prog, out = sys.argv[1], sys.argv[2]
sets = {n: EV.get_set(n, EV.NTR[n], FC.TRB, EV.R) for n in EV.TRAIN_N}
spec = importlib.util.spec_from_file_location("cand", prog); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
t = time.time(); predict = mod.train(EV.API(sets), EV.BUDGET); print("train", time.time() - t, flush=True)
n = 16; R = EV.R; D = EV.get_set(n, 100, FC.TEB, R); F = 100
fstate = jax.jit(lambda g, b_, w, pv, VP, BB, PERM, S, dS, dB: PC.circuit_state(n, g, b_, w, pv, VP, BB, PERM, S, n / D["m"], dS, dB))
kw = dict(place=EV.place_of(D, np.arange(F))) if "place" in inspect.signature(predict).parameters else {}
o = EV.check_outs(predict(n, D["V"].copy(), D["SG"].copy(), D["m"], **kw), F, n, D["m"])
P0 = np.zeros((F, 2**n), np.float32); E0 = np.zeros((F, 2**n), np.int8); prr = np.zeros(F)
for i in range(F):
    idx = np.array([i]); args = FC.entries(D, idx, {k: np.asarray(o[k])[idx] for k in ("gamma", "beta", "w", "pv")})
    dS, dB = EV._corr({k: (o[k][idx] if k in o else None) for k in ("dS", "dB")}, n)
    fin = np.asarray(fstate(*args[:8], dS, dB)[-1])
    m_, cl, E = sim.gen_instance(n, int(D["seed"][i])); E = np.asarray(E, np.float64)
    ps = []
    for r in range(R):
        Ec = CP.permute_vars(E, n, D["PERM"][i, r]); ps.append(fin[r][Ec == 0].sum())
        if r == 0: P0[i] = fin[0]; E0[i] = Ec
    prr[i] = np.mean(ps)
np.savez_compressed(out, P=P0, E=E0, prr=prr, seed=D["seed"]); print("saved", out, time.time() - t)

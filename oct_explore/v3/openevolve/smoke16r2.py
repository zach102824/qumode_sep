import sys, time, importlib.util, numpy as np
sys.path.insert(0, 'net'); import evaluator_net as EV, frozen_circuit as FC, jax, jax.numpy as jnp
prog = sys.argv[1]
spec = importlib.util.spec_from_file_location("c", prog); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
sets = {n: EV.get_set(n, 600, FC.TRB, 4) for n in (8, 12)}
t = time.time(); pr = mod.train(EV.API(sets), 25); print("train", time.time() - t)
D = EV.get_set(16, 60, FC.VAB, 4); D = {**D, **{k: D[k][:6] for k in ("V", "SG", "PERM", "S", "seed")}, "sol": D["sol"][:6]}
t = time.time(); r = EV.score_set(pr, D, 2); print("score16", r, time.time() - t)
rng = np.random.default_rng(0); p = mod.init(rng); p.update(mod.native_init(rng))
fx = mod.features(12, sets[12]["V"][:2], sets[12]["SG"][:2], sets[12]["m"]); pl = EV.place_of(sets[12], np.arange(2))
o0 = mod.apply(p, *fx, place=pl); p["gS"] = p["gS"] + 0.3; p["gB"] = p["gB"] + 0.3; o1 = mod.apply(p, *fx, place=pl)
print("dS change", float(jnp.abs(o1["dS"] - o0["dS"]).max()), "dB change", float(jnp.abs(o1["dB"] - o0["dB"]).max()))

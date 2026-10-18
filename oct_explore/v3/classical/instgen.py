"""Dump instances using the exact v3 generator (sim.instances). Run with JAX on CPU."""
import os, sys, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import numpy as np, sim
TRB, TEB = 31_000_000_000, 32_000_000_000
out = {}
spec = [("test", n, (200 if n == 14 else 100), TEB) for n in (8, 10, 12, 14, 20)] + [("train", 8, 200, TRB), ("train", 20, 50, TRB)]
for role, n, N, base in spec:
    L = []
    for ins in sim.instances(n, N, base):
        m, cl, E = sim.gen_instance(n, ins["seed"]); assert int((E == 0).sum()) == ins["nsol"]
        L.append(dict(seed=ins["seed"], m=m, nsol=ins["nsol"], cl=[[list(v), list(s)] for v, s in cl]))
    out[f"{role}_{n}"] = L; print(role, n, len(L), "nsol median", np.median([i["nsol"] for i in L]), flush=True)
json.dump(out, open(os.path.join(HERE, "instances.json"), "w"))

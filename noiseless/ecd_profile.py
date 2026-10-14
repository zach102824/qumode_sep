"""Instrumented profile of one shortened n=20 3cav trial: wraps the circuit pieces with timers (same math)."""
import json, os, sys, time
import numpy as np
import noiseless.relayout_3cav as r3
from noiseless.ecd_kbit import born, displace_fast
from noiseless.circuit_local_ecd import rotation_np

OUT = os.environ.get("PROF_OUT", "noiseless/results/ecd_speed/profile.json")
T = {k: 0.0 for k in ("rot", "ecd_build", "ecd_apply", "jp", "born", "cost_gibbs", "circuit_calls")}
T0 = time.perf_counter()

def dump():
    d = dict(T); d["wall_since_import"] = time.perf_counter() - T0; d["pid"] = os.getpid()
    json.dump(d, open(OUT + f".{os.getpid()}", "w"), indent=1)

def circ(x, n_layers, layout, u_name="jp", _cache={}):
    P = layout.P; h = r3.make_param_helpers(P)
    key = (layout.ks, layout.nfs, tuple(layout.edges), u_name)
    if key not in _cache:
        _cache[key] = r3.jp_phase(layout)
    ph = _cache[key]
    psi = np.zeros(layout.dims, dtype=complex); psi[(0,) * len(layout.dims)] = 1.0
    for betas, thetas, phis in h["unpack"](x, n_layers):
        t0 = time.perf_counter()
        for t in range(P):
            psi = r3._apply_axis(rotation_np(float(thetas[t]), float(phis[t])), psi, t)
        t1 = time.perf_counter(); T["rot"] += t1 - t0
        for t in range(P):
            a = time.perf_counter()
            D = displace_fast(complex(betas[t]) / 2.0, layout.nfs[t])
            b = time.perf_counter(); T["ecd_build"] += b - a
            p0 = np.take(psi, 0, axis=t); p1 = np.take(psi, 1, axis=t); ax = P + t - 1
            new1 = r3._apply_axis(D, p0, ax); new0 = r3._apply_axis(D.conj().T, p1, ax)
            psi = np.stack([new0, new1], axis=t)
            T["ecd_apply"] += time.perf_counter() - b
        t2 = time.perf_counter(); psi = psi * ph; T["jp"] += time.perf_counter() - t2
    return psi.reshape(-1)

def probs_from_x(self, x):
    k = circ(x, self.n_layers, self.layout, self.u_name)
    t = time.perf_counter(); p = born(k); T["born"] += time.perf_counter() - t
    T["circuit_calls"] += 1
    if int(T["circuit_calls"]) % 50 == 0: dump()
    return p
r3.XorMultiCavSim.probs_from_x = probs_from_x
_g = r3.gibbs_objective
def g(*a, **k):
    t = time.perf_counter(); v = _g(*a, **k); T["cost_gibbs"] += time.perf_counter() - t; return v
r3.gibbs_objective = g

if __name__ == "__main__":
    from noiseless import run_relayout_kbit
    sys.argv = ["run_relayout_kbit.py"] + sys.argv[1:]
    run_relayout_kbit.main(sys.argv[1:])

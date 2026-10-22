"""Check K-cavity generalisation. (a) K=4 (n=16): old F16_qaoa angles on old TEB formula #0 reproduce the stored p.
(b) K=5 (n=20): cavity-chain sim (nsim, 5x16-level, Gray one-bit-flip mixer) == independent qubit QAOA statevector (numpy, RX on each bit).
(c) K=5: frozen circuit_p (glob=True, w=0) == same qubit QAOA."""
import common as CM, json, numpy as np, jax.numpy as jnp, sim, compiler as CP, nsim as NS, frozen_circuit as FC
J = json.load(open(f"{CM.ROOT}/results/v3/native/fixed/F16_qaoa.json")); x = np.array(J["x"])
def chain_p(x, E, n):
    ks = CP.LAYOUTS[n]; nm = n / int(round(9.93 * n)); PE = jnp.asarray((E[None] * nm).astype(np.float32)); S = jnp.asarray((E[None] == 0).astype(np.float32))
    mx = [NS.Mixer(k, "gray") for k in ks]; psi = NS.uniform_psi0(ks, 1)
    psi = NS.run_state(psi, [x[2*l+1] * PE for l in range(4)], [[m(x[2*l]) for m in mx] for l in range(4)], ks)
    return float(jnp.sum(jnp.abs(psi.reshape(1, -1))**2 * S))
def qubit_p(x, E, n):
    nm = n / int(round(9.93 * n)); psi = np.full(1 << n, 2**(-n/2), np.complex128)
    for l in range(4):
        psi = psi * np.exp(-1j * x[2*l+1] * nm * E); b = x[2*l]; c, s = np.cos(b/2), -1j*np.sin(b/2)
        t = psi.reshape((2,)*n)
        for i in range(n):
            t0, t1 = np.take(t, 0, i), np.take(t, 1, i); t = np.stack([c*t0 + s*t1, s*t0 + c*t1], i)
        psi = t.reshape(-1)
    return float(np.sum(np.abs(psi)**2 * (E == 0)))
ins = sim.instances(16, 1, 32e9)[0]; print("(a) n=16 K=4 old formula: stored", J["test"]["16"]["p"][0], "now", chain_p(x, ins["E"].astype(np.float64), 16))
ins20 = sim.instances(20, 1, CM.TEB)[0]; E = ins20["E"].astype(np.float64)
print("(b) n=20 K=5 chain sim", chain_p(x, E, 20), "qubit QAOA", qubit_p(x, E, 20))
D = FC.make_set(20, 1, CM.TEB, 1); o = dict(gamma=np.tile(x[1::2], (1, 1)), beta=np.tile(x[0::2], (1, 1)), w=np.zeros((1, D["m"], 4)), pv=np.full((1, 20), .5))
a = FC.entries(D, [0], o); print("(c) n=20 K=5 frozen circuit_p glob", float(FC.circuit_p(20, *a[:8], a[8], glob=True)[0]))

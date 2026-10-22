"""unit tests for the net16x pair unitary: unitarity, gradient, H=0 identity / exact match with the frozen circuit."""
import os, sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frozen_circuit as FC, pair_circuit as PC, evaluator_net as EV
rng = np.random.default_rng(0); ok = True
def rep(name, cond, val): global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {name}: {val}")
A = rng.normal(size=(256, 256)) + 1j * rng.normal(size=(256, 256)); H = PC.herm(A.real * .05, A.imag * .05); U = np.asarray(PC.pair_unitary(H))
rep("unitarity |U^dag U - I|max", np.abs(U.conj().T @ U - np.eye(256)).max() < 1e-4, np.abs(U.conj().T @ U - np.eye(256)).max())
rep("H=0 -> U=I", np.abs(np.asarray(PC.pair_unitary(PC.herm(np.zeros((256, 256))))) - np.eye(256)).max() < 1e-7, 0)
for n in (8, 12, 16):
    D = EV.get_set(n, 600 if n < 16 else 60, FC.TRB if n < 16 else FC.VAB, 4); idx = np.arange(2); m = D["m"]; C = n // 4
    o = dict(gamma=rng.normal(size=(2, 4)) * .3, beta=rng.normal(size=(2, 4)), w=rng.normal(size=(2, m, 4)) * .1, pv=rng.uniform(.2, .8, size=(2, n)))
    args = FC.entries(D, idx, o)
    p0 = np.asarray(FC.circuit_p(n, *args)); p1 = np.asarray(PC.circuit_px(n, *args)); Z = np.zeros((4, 3, C - 1, 256, 256), np.float32)
    p2 = np.asarray(PC.circuit_px(n, *args, pair=dict(pHr=Z, pHi=Z)))
    rep(f"n={n} no-pair == frozen", np.abs(p0 - p1).max() < 1e-6, np.abs(p0 - p1).max()); rep(f"n={n} H=0 == frozen", np.abs(p0 - p2).max() < 1e-5, np.abs(p0 - p2).max())
    Hr = np.zeros_like(Z); Hr[1, 1, 0] = np.real(np.asarray(H)); Hi = np.zeros_like(Z); Hi[1, 1, 0] = np.imag(np.asarray(H))
    st = PC.circuit_state(n, *args, pair=dict(pHr=Hr, pHi=Hi)); rep(f"n={n} norm preserved", abs(float(st[-1].sum(1).max()) - 1) < 1e-4, float(st[-1].sum(1).max()))
    # swap-permutation generator: U = P (perm) via H = (pi/2)(I - P) for an involution P -> exp(-iH)= P up to phase
    f = jax.jit(lambda s: PC.circuit_px(n, *args, pair=dict(pHr=jnp.asarray(Z).at[0, 1, 0].set(s * Hr[1, 1, 0]), pHi=jnp.asarray(Z).at[0, 1, 0].set(s * Hi[1, 1, 0]))).sum())
    g = float(jax.grad(f)(1.0)); e = 1e-3; fd = (float(f(1.0 + e)) - float(f(1.0 - e))) / (2 * e)
    rep(f"n={n} grad vs finite-diff", np.isfinite(g) and abs(g - fd) < 2e-2 * max(1, abs(fd)), (g, fd))
    rep(f"n={n} count_pairs", PC.count_pairs(dict(pHr=Hr, pHi=Hi), C) == 1, PC.count_pairs(dict(pHr=Hr, pHi=Hi), C))
    # structured per-entry form: pc * pG equals the shared form
    Bn = args[0].shape[0]; pc = np.zeros((Bn, 4, 3, C - 1, 2), np.float32); pc[:, 1, 1, 0, 0] = 1.0
    pG = np.stack([Hr[1, 1, 0], np.zeros((256, 256), np.float32)]); pGi = np.stack([Hi[1, 1, 0], np.zeros((256, 256), np.float32)])
    p3 = np.asarray(PC.circuit_px(n, *args, pair=dict(pG=pG, pGi=pGi, pc=pc))); p4 = np.asarray(PC.circuit_px(n, *args, pair=dict(pHr=Hr, pHi=Hi)))
    rep(f"n={n} structured == shared", np.abs(p3 - p4).max() < 1e-5, np.abs(p3 - p4).max()); rep(f"n={n} count structured", PC.count_pairs(dict(pG=pG, pGi=pGi, pc=pc), C) == 1, PC.count_pairs(dict(pG=pG, pGi=pGi, pc=pc), C))
    # pslots + broadcast: slot subset (1,2) with A=1 equals full-slot form with the block tiled
    Hb = np.zeros((4, 2, 1, 256, 256), np.float32); Hb[2, 1, 0] = Hr[1, 1, 0]; Hib = np.zeros_like(Hb); Hib[2, 1, 0] = Hi[1, 1, 0]
    Hf = np.zeros_like(Z); Hf[2, 2, :] = Hr[1, 1, 0]; Hif = np.zeros_like(Z); Hif[2, 2, :] = Hi[1, 1, 0]
    p5 = np.asarray(PC.circuit_px(n, *args, pair=dict(pHr=Hb, pHi=Hib, pslots=(1, 2)))); p6 = np.asarray(PC.circuit_px(n, *args, pair=dict(pHr=Hf, pHi=Hif)))
    rep(f"n={n} pslots/broadcast == full", np.abs(p5 - p6).max() < 1e-5, np.abs(p5 - p6).max()); rep(f"n={n} count broadcast", PC.count_pairs(dict(pHr=Hb, pHi=Hib), C) == C - 1, PC.count_pairs(dict(pHr=Hb, pHi=Hib), C))
    # mixer conjugation: mid=P, post=P^dag with P = CNOT relabel is unitary-consistent (norm) and differs from no-pair
# permutation test: swapping cavity a<->b contents via a pair permutation is exactly realised
P = np.zeros((256, 256)); [P.__setitem__((16 * j + i, 16 * i + j), 1) for i in range(16) for j in range(16)]
Hp = (np.pi / 2) * (np.eye(256) - P); Up = np.asarray(PC.pair_unitary(PC.herm(Hp))); rep("swap perm realised", np.abs(np.abs(Up) - P).max() < 1e-4, np.abs(np.abs(Up) - P).max())
print("ALL PASS" if ok else "SOME FAIL"); sys.exit(0 if ok else 1)

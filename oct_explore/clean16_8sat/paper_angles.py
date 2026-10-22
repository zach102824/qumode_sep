"""Reference rows: the paper's PUBLISHED fixed angles (PhaseCraft qaoa_ksat_paper_data, key (8, 176.54)), no training,
evaluated on the same clean16_8sat test sets. Qubit QAOA: |+>^n, per layer exp(+i gamma E/2) then prod_j exp(-i beta X_j/2)
(sign convention verified in qaoa_ksat/README.md against their n=12 p=60 mean p_succ 0.56)."""
import sys, json, numpy as np, common as CM, sim
sys.path.insert(0, "/workspace/qumode_sep/qaoa_ksat"); from paper_data.optimal_angles import optimal_angles as A
p = int(sys.argv[1]); ang = A[(8, 176.54)][p]; out = {"p": p, "test": {}}
def psucc(E, n):
    psi = np.full(1 << n, 2 ** (-n / 2), np.complex128); Ef = E.astype(np.float64)
    for b, g in zip(ang["betas"], ang["gammas"]):
        psi = psi * np.exp(0.5j * g * Ef); c, s = np.cos(b / 2), -1j * np.sin(b / 2); t = psi.reshape((2,) * n)
        for i in range(n):
            t0, t1 = np.take(t, 0, i), np.take(t, 1, i); t = np.stack([c * t0 + s * t1, s * t0 + c * t1], i)
        psi = t.reshape(-1)
    return float(np.sum(np.abs(psi[E == 0]) ** 2))
for n in [("val", CM.VAL_N)] + CM.TEST_N:
    tag, n = n if isinstance(n, tuple) else (str(n), n)
    ins = sim.instances(n, CM.NVAL, CM.VAB) if tag == "val" else sim.instances(n, CM.NTEST, CM.TEB)
    pp = np.array([psucc(i["E"], n) for i in ins]); ip = 1 / np.maximum(pp, 1e-12)
    out["test"][tag] = dict(p=pp.tolist(), median_inv_p=float(np.median(ip)), B90_inv_p=float(np.quantile(ip, .9)), mean_p=float(pp.mean()))
    print(p, tag, out["test"][tag]["median_inv_p"], out["test"][tag]["B90_inv_p"], out["test"][tag]["mean_p"], flush=True)
    json.dump(out, open(f"{CM.RES}/P{p}.json", "w"))

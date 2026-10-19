"""Evaluate the trained QAOA n=6 L4/L8 (results/v3/sd_n6/QAOA_n6_L*.json, best-train candidate) on the 300 val set (33e9) for S comparison."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, sim
n = 6; vas = sim.instances(n, 300, 33_000_000_000)
x_ = np.arange(1 << n); X = np.array([[1, 0], [0, 1]])
def probs(x, insts):
    out = []
    for ins in insts:
        E = ins["E"].astype(float); cn = E * n / ins["m"]; psi = np.full(1 << n, 1 / 8.0, complex).reshape((2,) * n)
        for l in range(len(x) // 2):
            b, g = x[2 * l], x[2 * l + 1]; psi = psi * np.exp(-1j * g * cn).reshape(psi.shape)
            M = np.array([[np.cos(b / 2), -1j * np.sin(b / 2)], [-1j * np.sin(b / 2), np.cos(b / 2)]])
            for q in range(n): psi = np.moveaxis(np.tensordot(M, psi, axes=([1], [q])), 0, q)
        out.append(float(np.sum(np.abs(psi.reshape(-1)) ** 2 * (E == 0))))
    return np.array(out)
for L in (4, 8):
    f = f"results/v3/sd_n6/QAOA_n6_L{L}.json"; J = json.load(open(f)); p = probs(J["best"]["x"], vas); inv = 1 / p
    J["val"] = {"median_inv_p": float(np.median(inv)), "B90_inv_p": float(np.quantile(inv, 0.9)), "loss": float(np.mean(-np.log(p))),
                "S": float((np.log(np.median(inv)) + np.log(np.quantile(inv, 0.9))) / 2)}
    tp = probs(J["best"]["x"], sim.instances(n, 200, 32_000_000_000)); assert abs(np.median(1 / tp) - J["best"]["test"]["median_inv_p"]) < 1e-3 * np.median(1 / tp)
    J["best"]["test"]["S"] = float((np.log(J["best"]["test"]["median_inv_p"]) + np.log(J["best"]["test"]["B90_inv_p"])) / 2)
    json.dump(J, open(f, "w")); print(L, "val", J["val"], "test S", J["best"]["test"]["S"])

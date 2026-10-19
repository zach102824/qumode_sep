import json, sys, numpy as np
from scipy.stats import spearmanr
d = json.load(open(sys.argv[1])); n = sys.argv[2]; t = d["test"][n]
P = np.array(t["ps"]); F = np.array(t["feats"])  # (I, R+1), (I, R+1, 2)
def summ(p): ip = 1 / np.maximum(p, 1e-300); return f"{np.median(ip):.1f}/{np.percentile(ip, 90):.1f}"
print("single mincut", summ(P[:, 0]), "| round-robin all", summ(P.mean(1)), "| oracle", summ(P.max(1)))
for j, nm in enumerate(["uniform_resid", "bias_resid"]):
    rho = np.nanmean([spearmanr(F[i, :, j], -np.log(P[i]))[0] for i in range(len(P))])
    pick = P[np.arange(len(P)), F[:, :, j].argmin(1)]
    o = np.argsort(F[:, :, j], 1)
    tk = {k: summ(np.take_along_axis(P, o[:, :k], 1).mean(1)) for k in (2, 4)}
    print(f"{nm}: within-formula spearman(resid, -log p) {rho:.3f}; top1 {summ(pick)}; top2-rr {tk[2]}; top4-rr {tk[4]}")

"""Lookup-to-GS stats for SATLike on wmaxsat_conv (logs/conv/satlike_conv.jsonl), bootstrap std over INSTANCES.
Unsolved runs (hit<0) = inf. Percentiles: numpy 'higher' (an actual run value)."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = 1000; XS = [100, 250, 500, 1000, 2000]; Q = {"median": 50, "B90": 90, "B99": 99}
rows = [json.loads(l) for l in open("logs/conv/satlike_conv.jsonl")]
data = {}
for r in rows: data.setdefault(r["n"], {})[r["inst"]] = [h if h >= 0 else np.inf for h in r["hits"]]
def stats(H):  # H: (inst, runs)
    f = H.reshape(-1); return {k: float(np.percentile(f, q, method="higher")) for k, q in Q.items()}
def boot(H, rng):
    s = {k: [] for k in Q}
    for _ in range(R):
        st = stats(H[rng.integers(0, len(H), len(H))])
        for k in Q: s[k].append(st[k])
    return {k: float(np.std(v)) if np.all(np.isfinite(v)) else np.inf for k, v in s.items()}
rng = np.random.default_rng(12345); res = {}; conv = {}
for n in sorted(data):
    H = np.array([data[n][i] for i in sorted(data[n])]); res[n] = (len(H), int(np.isinf(H).sum()), H.size, stats(H), boot(H, rng))
    conv[n] = {x: (stats(H[:x]), boot(H[:x], rng)) for x in XS if x <= len(H)}
ns = np.array(sorted(res)); fits = {}
for k in Q:
    y = np.log([res[n][3][k] for n in ns]); A = np.polyfit(ns, y, 1); yp = np.polyval(A, ns)
    fits[k] = (np.exp(A[1]), np.exp(A[0]), 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum())
L = ["# SATLike convergence study (wmaxsat_conv)", "",
     "Instances: new unique-GS random weighted 4-SAT (m=round(9.93n), w in 1..5), same generator as wmaxsat_fresh, seeds 5e9+1e7n+k (disjoint). "
     "SATLike = strong_classical.runone, cfg BEST[satlike|n], cap min(40*2^n,4e6), 5 runs/instance. Strict lookups to first GS check. "
     "Unsolved = inf. Std = bootstrap over instances (1000 resamples, instance keeps its runs).", "",
     "| n | instances | runs | unsolved | median | B90 | B99 |", "|---|---|---|---|---|---|---|"]
for n in ns:
    I, u, T, s, e = res[n]; L.append(f"| {n} | {I} | {T} | {u} | " + " | ".join(f"{s[k]:.0f} ± {e[k]:.1f}" for k in Q) + " |")
L += ["", "## Fits a·b^n", "", "| stat | a | b | R² |", "|---|---|---|---|"] + [f"| {k} | {a:.3f} | {b:.4f} | {r2:.4f} |" for k, (a, b, r2) in fits.items()]
L += ["", "## Convergence vs number of instances X (value ± std)", ""]
for k in Q:
    L += [f"### {k}", "", "| n | " + " | ".join(f"X={x}" for x in XS) + " |", "|---" * (len(XS) + 1) + "|"]
    for n in ns: L.append(f"| {n} | " + " | ".join(f"{conv[n][x][0][k]:.0f} ± {conv[n][x][1][k]:.1f}" if x in conv[n] else "-" for x in XS) + " |")
    L.append("")
L += ["![](satlike_convergence.png)", ""]
open("noiseless/results/SATLIKE_CONVERGENCE.md", "w").write("\n".join(L))
plt.rcParams.update({"text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black", "font.size": 12})
fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor="white"); ax.set_facecolor("white")
for k, mk in zip(Q, "osD"):
    v = np.array([res[n][3][k] for n in ns]); e = np.array([res[n][4][k] for n in ns]); a, b, r2 = fits[k]
    l = ax.errorbar(ns, v, yerr=e, fmt=mk, capsize=3, label=f"{k}: {a:.2f}·{b:.3f}^n (R²={r2:.3f})")
    ax.plot(ns, a * b ** ns, "--", color=l[0].get_color(), lw=1)
ax.set_yscale("log"); ax.set_xlabel("n"); ax.set_ylabel("SATLike lookups to ground state"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("noiseless/results/satlike_convergence.png", dpi=150, facecolor="white")
print("\n".join(L[:30]))
print("SATLIKE_CONV = {" + ", ".join(f"{n}: {{" + ", ".join(f"'{k}': ({res[n][3][k]:.0f}, {res[n][4][k]:.1f})" for k in Q) + f", 'instances': {res[n][0]}, 'unsolved': {res[n][1]}}}" for n in ns) + "}")

"""Stats for unweighted 4-SAT SATLike study (logs/4sat/satlike_4sat.jsonl): median/B90/B99, bootstrap std over instances, convergence, fits, SAT/UNSAT split."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = 1000; XS = [100, 250, 500, 1000, 2000]; Q = {"median": 50, "B90": 90, "B99": 99}
rows = [json.loads(l) for l in open("logs/4sat/satlike_4sat.jsonl")]; cfg = json.load(open("logs/4sat/chosen_cfg.json"))
data, sat, att = {}, {}, {}
for r in rows:
    data.setdefault(r["n"], {})[r["inst"]] = [h if h >= 0 else np.inf for h in r["hits"]]; sat.setdefault(r["n"], {})[r["inst"]] = r["E_gs"] == 0
    att.setdefault(r["n"], []).append(r["attempts"])
def stats(H): f = H.reshape(-1); return {k: float(np.percentile(f, q, method="higher")) for k, q in Q.items()}
def boot(H, rng):
    s = {k: [] for k in Q}
    for _ in range(R):
        st = stats(H[rng.integers(0, len(H), len(H))])
        for k in Q: s[k].append(st[k])
    return {k: float(np.std(v)) if np.all(np.isfinite(v)) else np.inf for k, v in s.items()}
rng = np.random.default_rng(12345); res, conv, split = {}, {}, {}
for n in sorted(data):
    ids = sorted(data[n]); H = np.array([data[n][i] for i in ids]); S = np.array([sat[n][i] for i in ids])
    res[n] = (len(H), int(np.isinf(H).sum()), H.size, stats(H), boot(H, rng))
    conv[n] = {x: (stats(H[:x]), boot(H[:x], rng)) for x in XS if x <= len(H)}
    split[n] = {lab: (int(m.sum()), int(np.isinf(H[m]).sum()), stats(H[m]), boot(H[m], rng)) for lab, m in (("SAT", S), ("UNSAT", ~S))}
ns = np.array(sorted(res)); fits = {}
for k in Q:
    y = np.log([res[n][3][k] for n in ns]); A = np.polyfit(ns, y, 1); yp = np.polyval(A, ns)
    fits[k] = (np.exp(A[1]), np.exp(A[0]), 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum())
f = lambda s, e, k: f"{s[k]:.0f} ± {e[k]:.1f}"
L = ["# SATLike convergence study: unweighted random 4-SAT (4sat_conv)", "",
     "Instances: new unique-GS random 4-SAT, all clause weights 1 (otherwise same generator as wmaxsat_conv: 4 distinct vars/clause, m=round(9.93n)), "
     "seeds 1.1e10+1e7n+k (disjoint from every existing base). Both SAT (E_gs=0) and UNSAT kept. "
     f"SATLike = strong_classical.runone, ONE cfg for all n: {cfg['cfg']} (candidate {cfg['name']}, best avg B90 over the same 4 candidates on held-out 200 inst x2 at n=12,16,20; logs/4sat/uniform_tune.out). "
     "Cap min(40*2^n,4e6), 5 runs/instance, strict lookups to first GS check. Unsolved = inf. Std = bootstrap over instances (1000 resamples).", "",
     "| n | instances | runs | unsolved | median | B90 | B99 | unique-GS acceptance | SAT / UNSAT |", "|---|---|---|---|---|---|---|---|---|"]
for n in ns:
    I, u, T, s, e = res[n]; a = att[n]
    L.append(f"| {n} | {I} | {T} | {u} | " + " | ".join(f(s, e, k) for k in Q) + f" | {I}/{sum(a)} = {I/sum(a):.4f} | {split[n]['SAT'][0]} / {split[n]['UNSAT'][0]} |")
L += ["", "## Fits a·b^n", "", "| stat | a | b | R² |", "|---|---|---|---|"] + [f"| {k} | {a:.3f} | {b:.4f} | {r2:.4f} |" for k, (a, b, r2) in fits.items()]
L += ["", "## SAT vs UNSAT instances", "", "| n | class | instances | unsolved | median | B90 | B99 |", "|---|---|---|---|---|---|---|"]
for n in ns:
    for lab in ("SAT", "UNSAT"):
        c, u, s, e = split[n][lab]; L.append(f"| {n} | {lab} | {c} | {u} | " + " | ".join(f(s, e, k) for k in Q) + " |")
L += ["", "## Convergence vs number of instances X (value ± std)", ""]
for k in Q:
    L += [f"### {k}", "", "| n | " + " | ".join(f"X={x}" for x in XS) + " |", "|---" * (len(XS) + 1) + "|"]
    for n in ns: L.append(f"| {n} | " + " | ".join(f(*conv[n][x], k) if x in conv[n] else "-" for x in XS) + " |")
    L.append("")
L += ["## Weighted baseline (SATLIKE_CONV, commit 26769729) at n=20: median 221 ± 3.9, B90 871 ± 18.8, B99 2877 ± 161.8", "", "![](satlike_4sat.png)", ""]
open("noiseless/results/SATLIKE_4SAT.md", "w").write("\n".join(L))
plt.rcParams.update({"text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black", "font.size": 12})
fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor="white"); ax.set_facecolor("white")
for k, mk in zip(Q, "osD"):
    v = np.array([res[n][3][k] for n in ns]); e = np.array([res[n][4][k] for n in ns]); a, b, r2 = fits[k]
    l = ax.errorbar(ns, v, yerr=e, fmt=mk, capsize=3, label=f"{k}: {a:.2f}·{b:.3f}^n (R²={r2:.3f})")
    ax.plot(ns, a * b ** ns, "--", color=l[0].get_color(), lw=1)
ax.set_yscale("log"); ax.set_xlabel("n"); ax.set_ylabel("SATLike lookups to ground state (4-SAT)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("noiseless/results/satlike_4sat.png", dpi=150, facecolor="white")
print("\n".join(L))
print("SATLIKE_4SAT = {" + ", ".join(f"{n}: {{" + ", ".join(f"'{k}': ({res[n][3][k]:.0f}, {res[n][4][k]:.1f})" for k in Q) + f", 'instances': {res[n][0]}, 'unsolved': {res[n][1]}, 'sat': {split[n]['SAT'][0]}}}" for n in ns) + "}")

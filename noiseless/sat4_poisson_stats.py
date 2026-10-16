"""Stats for sat4_poisson_run.py (logs/4sat_pois/satlike_4sat_pois.jsonl) vs fixed-m unique-GS baseline (SATLIKE_4SAT.md, commit e3740317)."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = 1000; Q = {"median": 50, "B90": 90, "B99": 99}
FU = {8: (30, 103, 232), 10: (45, 161, 347), 12: (67, 229, 504), 14: (93, 321, 752), 16: (122, 428, 980), 18: (154, 572, 1475)}
FUFIT = {"median": (9.723, 1.1669), "B90": (32.241, 1.1721), "B99": (61.734, 1.1889)}
NAME = {"fu": "F-unique (ours, fixed m, unique GS)", "pu": "P-unique (Poisson m, unique GS)", "ps": "P-sat (Poisson m, all SAT, any solution; paper)", "fs": "F-sat (fixed m, all SAT, any solution)"}
rows = [json.loads(l) for l in open("logs/4sat_pois/satlike_4sat_pois.jsonl")]
D = {}
for r in rows: D.setdefault((r["arm"], r["n"]), []).append(r)
def stats(H): f = H.reshape(-1); return {k: float(np.percentile(f, q, method="higher")) for k, q in Q.items()}
rng = np.random.default_rng(12345)
def boot(H):
    s = {k: [] for k in Q}
    for _ in range(R):
        st = stats(H[rng.integers(0, len(H), len(H))])
        for k in Q: s[k].append(st[k])
    return {k: float(np.std(v)) for k, v in s.items()}
res = {}
for (a, n), rr in sorted(D.items()):
    rr.sort(key=lambda r: r["inst"]); H = np.array([[h if h >= 0 else np.inf for h in r["hits"]] for r in rr], float)
    att = sum(r["attempts"] for r in rr)
    res[(a, n)] = dict(I=len(rr), uns=int(np.isinf(H).sum()), s=stats(H), e=boot(H), rej=1 - len(rr) / att,
                       m=np.mean([r["m"] for r in rr]), nsol=np.median([r["nsol"] for r in rr]), sat=np.mean([r["E_gs"] == 0 for r in rr]))
ns = sorted({n for _, n in D}); arms = ["pu", "ps", "fs"]; fits = {}
for a in arms:
    for k in Q:
        x = np.array(ns); y = np.log([res[(a, n)]["s"][k] for n in ns]); A = np.polyfit(x, y, 1); fits[(a, k)] = (np.exp(A[1]), np.exp(A[0]))
for k in Q: fits[("fu", k)] = FUFIT[k]
L = ["# SATLike on unweighted random 4-SAT: Poisson m vs fixed m, unique-GS vs all-SAT", "",
     "Same generator as sat4_conv_run.py (4 distinct sorted vars, random signs, r=9.93), same SATLike cfg (logs/4sat/chosen_cfg.json: mode=0 swt=20 sp=0.01 t_bms=3), "
     "cap min(40·2^n,4e6), 5 runs/instance, 2000 instances per n per arm, strict lookups to first hit. Std = bootstrap over instances (1000 resamples). Script: noiseless/sat4_poisson_run.py, stats: noiseless/sat4_poisson_stats.py.", "",
     "- **F-unique**: our existing study (SATLIKE_4SAT.md, commit e3740317): m=round(9.93n), unique ground state (SAT or UNSAT), hit = that string.",
     "- **P-unique**: m ~ Poisson(9.93n), same unique-GS filter. Instance seeds 2.1e10+1e7n+k.",
     "- **P-sat** (PRX Quantum 5, 030348 style): m ~ Poisson(9.93n), keep satisfiable instances only, hit = any satisfying string. Seeds 2.2e10+1e7n+k.",
     "- **F-sat**: m=round(9.93n), keep satisfiable, hit = any satisfying string (separates the filter effect from the Poisson effect). Seeds 2.3e10+1e7n+k.", "",
     "| n | arm | instances | unsolved | rejection | mean m | median #solutions | SAT frac | median | B90 | B99 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for n in ns:
    L.append(f"| {n} | F-unique | 2000 | 0 | - | {round(9.93*n)} | 1 | - | " + " | ".join(str(v) for v in FU[n]) + " |")
    for a in arms:
        r = res[(a, n)]
        L.append(f"| {n} | {NAME[a].split(' (')[0]} | {r['I']} | {r['uns']} | {r['rej']:.3f} | {r['m']:.1f} | {r['nsol']:.0f} | {r['sat']:.3f} | " + " | ".join(f"{r['s'][k]:.0f} ± {r['e'][k]:.1f}" for k in Q) + " |")
L += ["", "## Fits a·b^n (n=8–18)", "", "| arm | median | B90 | B99 |", "|---|---|---|---|"]
for a in ["fu"] + arms: L.append(f"| {NAME[a].split(' (')[0]} | " + " | ".join(f"{fits[(a,k)][0]:.3f}·{fits[(a,k)][1]:.4f}^n" for k in Q) + " |")
L += ["", "## Ratio F-unique / arm (>1 means our set is harder)", "", "| n | vs P-unique med / B90 | vs P-sat med / B90 | vs F-sat med / B90 |", "|---|---|---|---|"]
RAT = {}
for n in ns:
    c = []
    for a in arms:
        rm = FU[n][0] / res[(a, n)]["s"]["median"]; rb = FU[n][1] / res[(a, n)]["s"]["B90"]; RAT[(a, n)] = (rm, rb); c.append(f"{rm:.2f} / {rb:.2f}")
    L.append(f"| {n} | " + " | ".join(c) + " |")
L += ["", "![](satlike_4sat_poisson.png)", ""]
open("noiseless/results/SATLIKE_4SAT_POISSON.md", "w").write("\n".join(L))
plt.rcParams.update({"text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black", "font.size": 11})
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white"); x = np.array(ns)
for ax, k, j in zip(axs, ["median", "B90"], [0, 1]):
    ax.set_facecolor("white")
    ax.plot(x, [FU[n][j] for n in ns], "ko-", label=NAME["fu"])
    for a, mk, c in zip(arms, "s^D", ["tab:blue", "tab:red", "tab:green"]):
        ax.errorbar(x, [res[(a, n)]["s"][k] for n in ns], yerr=[res[(a, n)]["e"][k] for n in ns], fmt=mk + "-", color=c, capsize=3, label=NAME[a])
    ax.set_yscale("log"); ax.set_xlabel("n"); ax.set_ylabel(f"SATLike lookups ({k})")
axs[0].legend(frameon=False, fontsize=8.5); fig.tight_layout(); fig.savefig("noiseless/results/satlike_4sat_poisson.png", dpi=150, facecolor="white")
print("\n".join(L))

"""Make results table (markdown) + scaling plot from results/*.json."""
import json, glob, sys
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SL = {8: (16, 70), 10: (26, 111), 12: (36, 157), 14: (50, 208), 16: (65, 272), 18: (82, 363)}
def sl(n): return SL.get(n, (4.931 * 1.1743 ** n, 21.372 * 1.1735 ** n))
tags = sys.argv[1:] or ["T_raw", "T_std", "C_raw", "QAOA_raw"]
rows = ["| run | L | n | median 1/p | B90 1/p | random median | SATLike median/B90 | median vs SATLike |", "|---|---|---|---|---|---|---|---|"]
plt.figure(figsize=(6, 4.2), facecolor="white")
for t in tags:
    try: r = json.load(open(f"results/{t}.json"))
    except FileNotFoundError: continue
    for n, d in sorted(r.get("test", {}).items(), key=lambda kv: int(kv[0])):
        for L, s in d.items():
            a, b = sl(int(n))
            rows.append(f"| {t} | {L} | {n} | {s['median_inv_p']:.1f} | {s['B90_inv_p']:.1f} | {s['rand_median']:.0f} | {a:.0f}/{b:.0f} | {s['median_inv_p']/a:.2f}x |")
    Lmax = max(int(L) for L in r["train"])
    ns = sorted(int(n) for n in r.get("test", {}) if str(Lmax) in r["test"][n] or Lmax in r["test"][n])
    if ns: plt.semilogy(ns, [r["test"][str(n) if str(n) in r["test"] else n][str(Lmax)]["median_inv_p"] for n in ns], "o-", label=f"{t} L={Lmax}")
ns = np.arange(8, 21); plt.semilogy(ns, [sl(n)[0] for n in ns], "k--", label="SATLike median")
plt.semilogy(ns, 2.0 ** ns / 2.5, "k:", label="random guess (approx)")
plt.xlabel("n"); plt.ylabel("median shots per success 1/p"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("results/scaling.png", dpi=150, facecolor="white")
open("results/table.md", "w").write("\n".join(rows) + "\n"); print("\n".join(rows))

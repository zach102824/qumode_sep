import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = np.load(sys.argv[1]); P, E, prr, seed = d["P"].astype(np.float64), d["E"], d["prr"], d["seed"]; tag = sys.argv[2]
F, N = P.shape; P /= P.sum(1, keepdims=True)
p0 = np.array([P[i][E[i] == 0].sum() for i in range(F)])
order = np.argsort(p0); worst = order[:10]; med = order[F//2-2:F//2+3]
pc = np.array([bin(x).count("1") for x in range(256)], np.int8)
def hd(a, b): x = a ^ b; return pc[x & 255] + pc[(x >> 8) & 255]
ids = np.arange(N)
def stats(i):
    p, e = P[i], E[i]; gs = np.flatnonzero(e == 0); H = np.min([hd(ids, g) for g in gs], 0)
    j = int(np.argmax(p)); s = np.sort(p)[::-1]; nz = p[p > 0]
    mE = {int(k): float(p[e == k].sum()) for k in range(min(int(e.max()), 6) + 1)}; mE["7+"] = float(p[e >= 7].sum())
    mH = {int(k): float(p[H == k].sum()) for k in range(17)}
    return dict(i=int(i), seed=int(seed[i]), p_gs=float(p0[i]), inv_p=float(1/p0[i]), inv_p_rr=float(1/prr[i]), n_gs=len(gs),
        max_p=float(p[j]), max_E=int(e[j]), max_H=int(H[j]), top10=float(s[:10].sum()), top100=float(s[:100].sum()),
        PR=float(1/np.sum(p**2)), S_bits=float(-(nz*np.log2(nz)).sum()), mass_E=mE, mass_H=mH,
        uni_E={k: float((e == k).mean()) for k in range(4)}, meanH=float((p*H).sum()), uniH=float(H.mean()), sorted=s)
W = [stats(i) for i in worst]; M = [stats(i) for i in med]
fmt = lambda x: f"{x:.3g}"
L = [f"| idx | 1/p (r=0) | #GS | max p (E, Ham) | top10 | PR (of 65536) | S bits (/16) | mass E=0/1/2/3/≥4 | uniform E=1/2 | <Ham> vs uniform | mass Ham 0-2/3-5/6-9/10+ |", "|"+"---|"*11]
for grp, rows in (("worst", W), ("median", M)):
    L.append(f"| **{grp}** |||||||||||")
    for r in rows:
        mE = r["mass_E"]; ge4 = 1 - sum(mE.get(k, 0) for k in range(4)); mh = r["mass_H"]
        hb = [sum(mh[k] for k in a) for a in (range(0,3), range(3,6), range(6,10), range(10,17))]
        L.append(f"| {r['i']} | {r['inv_p']:.0f} | {r['n_gs']} | {fmt(r['max_p'])} ({r['max_E']}, {r['max_H']}) | {r['top10']:.3f} | {r['PR']:.0f} | {r['S_bits']:.1f} | "
                 + "/".join(f"{mE.get(k,0):.3f}" for k in range(4)) + f"/{ge4:.3f} | {r['uni_E'][1]:.4f}/{r['uni_E'][2]:.4f} | {r['meanH']:.1f} vs {r['uniH']:.1f} | " + "/".join(f"{x:.2f}" for x in hb) + " |")
open(f"runs/hist_{tag}_table.md", "w").write("\n".join(L) + "\n")
json.dump([{k: v for k, v in r.items() if k != "sorted"} for r in W + M], open(f"runs/hist_{tag}.json", "w"), indent=1)
agg = lambda rows, k: np.median([r[k] for r in rows])
print("\n".join(L)); print("median worst PR", agg(W,"PR"), "top10", agg(W,"top10"), "S", agg(W,"S_bits"), "| median-group PR", agg(M,"PR"), agg(M,"top10"))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); sel = [(W[0], "worst #1"), (W[1], "worst #2"), (M[2], "median")]
for r, lab in sel: ax[0].loglog(np.arange(1, N+1), np.maximum(r["sorted"], 1e-12), label=f"{lab} idx{r['i']} 1/p={r['inv_p']:.0f} PR={r['PR']:.0f}")
ax[0].axhline(1/N, ls=":", c="k", label="uniform 1/65536"); ax[0].set_ylim(1e-9, 1); ax[0].set_xlabel("state rank"); ax[0].set_ylabel("prob"); ax[0].legend(fontsize=7); ax[0].set_title(f"{tag}: sorted state probabilities (r=0)")
ks = list(range(7)); wdt = 0.27
for j, (r, lab) in enumerate(sel):
    v = [r["mass_E"].get(k, 0) for k in range(6)] + [1 - sum(r["mass_E"].get(k, 0) for k in range(6))]
    ax[1].bar(np.array(ks) + (j-1)*wdt, v, wdt, label=f"{lab} idx{r['i']}")
ax[1].set_xticks(ks); ax[1].set_xticklabels([str(k) for k in range(6)] + ["6+"]); ax[1].set_xlabel("energy = # violated clauses"); ax[1].set_ylabel("prob mass"); ax[1].legend(fontsize=7); ax[1].set_title("mass by energy")
plt.tight_layout(); plt.savefig(f"FAILURE_HISTOGRAM_{tag}.png", dpi=120); print("png")

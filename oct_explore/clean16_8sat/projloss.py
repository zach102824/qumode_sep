"""Why the native (chain-projected) cost fails at k=8: per n, over 20 test formulas (block assignment, cavity = 4 consecutive vars):
fraction of clauses whose literals lie in one cavity / one neighbour pair (exactly representable by SNAP+bus), variance of E kept by the
chain projection, corr(PE, E), and random-guess cost 2^n/nsol."""
import numpy as np, common as CM, sim, compiler as CP, json
out = {}; L = ["| n | cavities | clauses in 1 cavity or a neighbour pair | var(E) kept by projection | corr(projected E, E) | P(argmin of projected E is a solution) | random guess 2^n/nsol (median) |", "|---|---|---|---|---|---|---|"]
for n in CM.TEST_N:
    ins = sim.instances(n, 20, CM.TEB); ks = CP.LAYOUTS[n]; fr, vk, cr, hit, rg = [], [], [], [], []
    for i in ins:
        m, cl, E = sim.gen_instance(n, i["seed"]); c = [set(v // 4 for v in vs) for vs, ss in cl]
        fr.append(np.mean([max(s) - min(s) <= 1 for s in c]))
        E = E.astype(np.float64); PE = CP.project(E[None], ks, "chain")[0]
        d, pd = E - E.mean(), PE - PE.mean(); vk.append((pd @ pd) / (d @ d)); cr.append((pd @ d) / np.sqrt((pd @ pd) * (d @ d)))
        hit.append(float(E[np.argmin(PE)] == 0)); rg.append(2 ** n / i["nsol"])
    r = dict(frac_local=float(np.mean(fr)), var_kept=float(np.mean(vk)), corr=float(np.mean(cr)), argmin_hit=float(np.mean(hit)), rand=float(np.median(rg))); out[n] = r
    L.append(f"| {n} | {len(ks)} | {r['frac_local']:.3f} | {r['var_kept']:.3f} | {r['corr']:.3f} | {r['argmin_hit']:.2f} | {r['rand']:.3g} |"); print(L[-1], flush=True)
json.dump(out, open(f"{CM.RES}/projloss.json", "w")); open(f"{CM.HERE}/PROJLOSS.md", "w").write("\n".join(L) + "\n")

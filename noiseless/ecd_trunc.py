"""(3) Truncation check on the 10 sl3 n=20 final parameter sets: top-5 Fock-level probability per cavity and
p(GS) change when the cutoffs grow 80/80/48 -> 96/96/64 (same parameters, same relabel mask)."""
import glob, json
import numpy as np
from noiseless.ecd_bench import apply_circuit_multi_fast2
from noiseless.ecd_kbit import born
from noiseless.relayout_3cav import MultiCavLayout, make_param_helpers
f = glob.glob("noiseless/results/relayout_runs/ovn_sl3/n20*.jsonl")[0]
recs = [json.loads(l) for l in open(f) if l.strip()]
h = make_param_helpers(3); out = []
def pgs(x, L, xa, gs, nfs):
    lay = MultiCavLayout((6, 6, 5), nfs, encoding="gray")
    p = born(apply_circuit_multi_fast2(np.asarray(x), L, lay))
    g = int(lay.flat_of_logical()[int(gs, 2) ^ xa])
    pd = p.reshape(lay.dims)
    marg = [pd.sum(axis=tuple(a for a in range(6) if a != 3 + c)) for c in range(3)]
    return float(p[g]), [float(m[-5:].sum()) for m in marg]
for r in recs:
    x = np.asarray(r["x"]); L = len(x) // 12; xa = int(r["sel_xa"])
    b = np.abs(h["betas_from_x"](x, L))
    p0, top0 = pgs(x, L, xa, r["ground_bitstring"], (80, 80, 48))
    p1, top1 = pgs(x, L, xa, r["ground_bitstring"], (96, 96, 64))
    row = {"inst": r["inst"], "L": L, "max_abs_beta": float(b.max()), "p_gs_logged": r["p_gs"], "p_gs_80_80_48": p0,
           "p_gs_96_96_64": p1, "dp_gs": p1 - p0, "top5_prob_80_80_48": top0, "top5_prob_96_96_64": top1}
    out.append(row); print(row, flush=True)
    json.dump(out, open("noiseless/results/ecd_speed/truncation.json", "w"), indent=1)

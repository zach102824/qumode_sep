import json, numpy as np
d = json.load(open("v3/classical/final_raw.json"))
def rk(x): return np.argsort(np.argsort(x))
def sp(a, b): return float(np.corrcoef(rk(a), rk(b))[0, 1])
Q = {"GNN-SNAP_L4 (S_g4_snapF_s1)": "S_g4_snapF_s1", "GNN-SNAP_L8 (S_g8_snapF_A_f)": "S_g8_snapF_A_f"}
quantum = {}; corr = {}
for k, t in Q.items():
    w = json.load(open(f"results/v3/{t}.json"))["test"]["winner"]
    quantum[k] = {n: {"median_inv_p": w[n]["median_inv_p"], "B90_inv_p": w[n]["B90_inv_p"]} for n in ["8", "10", "12", "14", "20"]}
    inv = 1 / np.array(w["20"]["p"]); corr[k] = {"vs_walksat_flips": sp(inv, d["20"]["walksat_p0.5"]["flips_per_inst"]),
        "vs_random_guess(1/nsol)": sp(inv, d["20"]["random_guess"]["per_inst"]), "vs_SA_n20train_evals": sp(inv, d["20"]["SA_tuned_n20train"]["evals_per_inst"])}
quantum["QAOA_L4 (LOG G_qfe table, same TEB set)"] = {"8": [5.3, 9.9], "10": [8.4, 19.7], "12": [12.8, 32.8], "14": [20.8, 59.6], "20": [84.9, 238.1]}
quantum["QAOA_L8 (LOG G_qfe table, same TEB set)"] = {"8": [2.5, 4.5], "10": [3.5, 7.3], "12": [4.6, 15.5], "14": [6.3, 18.3], "20": [16.4, 54.2]}
out = {"description": "Classical baselines on the exact v3 TEST instances: sim.instances(n, N, TEB=32e9), N=200 at n=14 else 100 (seeds listed). "
       "Target = all satisfying assignments (E==0), same success set as v3core.succ_D, so 1/p of the quantum circuit and classical costs share the target. "
       "Per-instance value = mean over 200 independent runs (expected cost to FIRST hit a satisfying string); then median / B90 (90th pct) across instances.",
       "units": {"quantum": "1/p = expected circuit shots (each shot = 1 sample, + 1 classical check)",
                 "random_guess": "expected uniform samples = 2^n / nsol (each = 1 energy eval)",
                 "flips": "number of single-variable flips (accepted moves) until first hit",
                 "evals": "WalkSAT: 1 initial + 4 break-count evaluations per step (one per variable of the chosen unsat clause); SA: 1 per random restart + 1 per Metropolis proposal (delta-energy)"},
       "settings": {"walksat": "SKC WalkSAT: pick random unsat clause; if some var has break 0 flip it (freebie), else w.p. p random var in clause, else min-break var. p=0.5 (best of {0.1..0.7} on n=20 TRAIN set; n=8 train is degenerate and prefers p=0.7, near random walk). No restarts (cap 2e8 never hit).",
                    "SA_tuned_n8train": "single-flip Metropolis, random var proposal, beta linear 1.0->2.0 over 5n proposals per restart, restart until hit; tuned on 200 n=8 TRAIN instances (grid b0{.2,.5,1}xb1{2,3,5}xsteps{5,20,50,200}n), criterion (log med+log B90)/2 of evals",
                    "SA_tuned_n20train": "same, beta 1.0->3.0 over 50n proposals; tuned on 50 n=20 TRAIN instances (base 31e9, disjoint from test); flat optimum (scores 7.44-7.47 for several settings)",
                    "runs_per_instance": 200, "instances_file": "v3/classical/instances.json", "code": "v3/classical/cls.c, runcls.py, tune.py, final.py"},
       "classical": {n: {k: ({"evals": v["stats"]} if k == "random_guess" else {"flips": v["flips_stats"], "evals": v["evals_stats"]}) for k, v in dd.items() if isinstance(v, dict)} for n, dd in d.items()},
       "classical_per_instance": {n: {k: v for k, v in dd.items()} for n, dd in d.items()},
       "quantum_ref": quantum, "spearman_n20_per_instance": corr}
json.dump(out, open("results/v3/classical_baselines.json", "w"), indent=1)
print(json.dumps(corr, indent=1))

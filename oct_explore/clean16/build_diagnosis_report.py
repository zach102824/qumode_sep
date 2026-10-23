import json, os
import numpy as np

ROOT = "/workspace/qumode_sep/oct_explore"
d4 = json.load(open(os.path.join(ROOT, "results/clean16/diagnosis_data.json")))
d8 = json.load(open(os.path.join(ROOT, "results/clean16_8sat/diagnosis_data.json")))

def q90(arr):
    return float(np.quantile(np.asarray(arr, dtype=np.float64), 0.90))

def med(arr):
    return float(np.median(np.asarray(arr, dtype=np.float64)))

def mean(arr):
    return float(np.mean(np.asarray(arr, dtype=np.float64)))

lines = []
A = lines.append

A("# Wavefunction Failure Diagnosis: Why the 16-Level Native Ansatz Fails on 4-SAT B90 and 8-SAT")
A("")
A("This report analyzes the full final and intermediate 2ⁿ-state wavefunctions across all 100 unseen test instances at n = 8, 12, 16, 20 for both **4-SAT** (`results/clean16/diagnosis_data.json`) and **8-SAT** (`results/clean16_8sat/diagnosis_data.json`), comparing:")
A("- **QAOA L4 (`Q`)**: All-to-all exact problem Hamiltonian H_C with 4 layers (8 global parameters).")
A("- **Global Cost L4 (`G`)**: All-to-all clause-weighted H_C + GNN per-layer parameters.")
A("- **Pure Chain-Projected QAOA L4 (`QC_block`, `QC_r0`, `QC_RR`)**: Exact 1D cavity-chain projection Proj_chain(E) = E₁ + E₂ − E_ov with no neural network (`QC_block` = identity variable order; `QC_r0` = min-cut placement; `QC_RR` = R=4 round-robin).")
A("- **Native `ckpt_110` (`N_r0`, `N_RR`)**: Full OpenEvolve winner (`Proj_pv(E_w) + corr(dS, dB) + grover(dS)`).")
A("- **Component Ablations of `ckpt_110` (at r=0)**:")
A("  - `N_pv_r0`: `Proj_pv(E_w)` only (pv-weighted chain projection of GNN-weighted clauses; `dS = 0, dB = 0`).")
A("  - `N_corr_r0`: `Proj_pv(E_w) + corr(dS, dB)` (without Grover phase boost).")
A("  - `N_grover_r0`: `Proj_pv(E_w) + grover(dS)` (without learned residual `dS, dB` corrections).")
A("  - `N_unif_dSdB`: Uniform projection `Proj_unif(E_w) + dS + dB` (replacing learned `pv` with `pv = 0.5`).")
A("  - `N_unif_w`: Uniform projection `Proj_unif(E_w)` only (`pv = 0.5, dS = 0, dB = 0`).")
A("")
A("---")
A("")
A("## Executive Summary: Locking vs. Smearing")
A("")
A("The wavefunction diagnostics reveal **two distinct physical failure regimes**—one in **4-SAT** driven by **over-concentration (locking into false attractors / ghost minima)**, and a second in **8-SAT** driven by **catastrophic projection collapse (71% Hamiltonian variance lost at n=20) combined with GNN cue failure**:")
A("")
A("1. **4-SAT Failure Mode — Hyper-Concentration into Ghost Minima (`N_eff` is 4.6× SMALLER than QAOA at n=20):**")
A("   - On 4-SAT, Native `ckpt_110` does **not** smear out. At n = 20 (2²⁰ = 1,048,576 states), All-to-All QAOA L4 has median participation ratio `N_eff = 2,836.3` (`H(P)/n = 0.737`), whereas Native `ckpt_110` (`N_r0`) collapses the wavefunction to **`N_eff = 619.7`** (`H(P)/n = 0.612`)—**4.6× more concentrated** than QAOA.")
A("   - Native concentrates **27.3% of the entire wavefunction** into the top 100 wrong bitstrings (`P_top100_wrong = 0.2728` vs QAOA's `0.1230`) and **8.02%** into just the top 10 wrong bitstrings (`P_top10_wrong = 0.0802` vs QAOA's `0.0259`).")
A("   - On the **10 hardest formulas (Hard10 / B90 tail)** at n = 20, Native still packs `7.65%` of all probability mass into 10 wrong bitstrings (`N_eff = 856.6`), but the true solution probability `p_sol` is starved (`P_top10_wrong / p_sol = 80,572×` vs `8.38×` for QAOA).")
A("   - **Why it locks onto the wrong state:** Even with min-cut placement (`r=0`), the 1D chain projection drops `35.9%` of the 4-SAT energy variance at n = 20 (`VarKept = 0.6414`), creating a median of **`275` uniform ghost minima** (mean `3,728` wrong bitstrings whose projected energy is strictly lower than the true satisfying assignment). While the GNN's bit-marginal prior `pv` (`76.4%` bit accuracy on 4-SAT) cuts the median ghost-minimum count tenfold (`275 → 28.5`, reducing `Med 1/p` from `1,675.9` to `342.6`), whenever `pv` mispredicts a cluster of `~3.7–4.7` variables on a hard instance, `Proj_pv(E_w)` + `dS/dB` + Grover amplifies a false attractor (`E_top1 = 2.23` violated clauses, `PE_z = -3.37 σ`) that beats the true ground state in projected energy **74% of the time**.")
A("")
A("2. **8-SAT Failure Mode — Projection Annihilation + Smearing Across 314,555 Ghost Minima + Broken GNN Cues:**")
A("   - Because an 8-SAT clause spans 8 variables across `C = 3, 4, 5` cavities, any clause touching 3 or more cavities (or non-adjacent cavities) has **zero** exact projection onto the 1D nearest-neighbor chain. At n = 20 (`C = 5`), the chain projection destroys **70.8% of the Hamiltonian variance** (`VarKept = 0.2918`, `Corr(Proj(E), E) = 0.540`), leaving `argmin Proj(E)` as a true solution **0.0% of the time** (its mean true energy is `E = 7.10` violated clauses) and creating **`6,712` median uniform ghost minima** (mean `37,129`).")
A("   - Simultaneously, on 8-SAT (`m ≈ 176.5 n` clauses) the GNN's bit-marginal predictions `pv` collapse to **56%–59% bit accuracy** (`pv_conf = 0.20`, barely above random 50% coin flips), and the Grover top-quantile mark (`mark > 0.5`) hits the true solution's cavity level **0.0% to 0.2% of the time**.")
A("   - Consequently, at n = 20 8-SAT the wavefunction is **both 3.4× more smeared out** than QAOA (`Med N_eff = 17,638` for `N_r0` and `15,694` for `QC_r0` vs `5,190` for QAOA; `97.6%` of all probability mass sits at `E ≥ 5` violated clauses) **and** every state it amplifies is a high-energy ghost minimum (`E_top1 = 6.78` violated clauses, `PE_z = -3.50 σ`, beating the true solution in projected energy **93% of the time**). Moreover, adding `ckpt_110`'s `dS/dB` tables on 8-SAT **hurts** performance (`N_r0` median `1/p = 96,305` vs `N_pv_r0` `61,529` and pure `QC_r0` `50,128`), exploding median effective ghost minima to **314,555** (30% of the entire Hilbert space) and causing Layer 4 at n = 16 to actively **decrease** `p_sol` (`0.00026 → 0.00019`).")
A("")
A("---")
A("")
A("## 1. Concentration vs. Smearing Across n = 8, 12, 16, 20")
A("")
A("Metrics over the 100 test instances (`B90 1/p` = 90th percentile of `1/p`, matching `OCT23_DATA.md`; `Hard10` = median over the 10 hardest instances with highest `1/p` for that method):")
A("")

for label, d in [("4-SAT", d4), ("8-SAT", d8)]:
    A(f"### 1.{1 if label=='4-SAT' else 2} {label}: Wavefunction Concentration (`H(P)/n`, `N_eff`) and False-Peak Mass (`P_top10_wrong`)")
    A("")
    A("| n | Method | Med 1/p | B90 1/p (Q90) | Med H(P)/n | Med N_eff (All / Hard10) | Med P_top1_wrong | Med P_top10_wrong (All / Hard10) | Med P_top100_wrong | Med Ratio P_top10_w / p_sol (All / Hard10) |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for n_str in ["8", "12", "16", "20"]:
        n = int(n_str)
        rows = d[n_str]
        for cname, mlabel in [
            ("Q", "QAOA L4 (`Q`)"),
            ("G", "Global Cost L4 (`G`)"),
            ("QC_r0", "Chain-Proj QAOA (`QC_r0`)"),
            ("N_pv_r0", "Native `Proj_pv` only (`N_pv_r0`)"),
            ("N_r0", "Native `ckpt_110` (`N_r0`)"),
        ]:
            invp = [r["circuits"][cname]["inv_p"] for r in rows]
            ent_n = [r["circuits"][cname]["ent"] / n for r in rows]
            neff = [r["circuits"][cname]["neff"] for r in rows]
            t1w = [r["circuits"][cname]["p_top1_wrong"] for r in rows]
            t10w = [r["circuits"][cname]["p_top10_wrong"] for r in rows]
            t100w = [r["circuits"][cname]["p_top100_wrong"] for r in rows]
            rat = [r["circuits"][cname]["p_top10_wrong"] / max(r["circuits"][cname]["p_sol"], 1e-12) for r in rows]
            idx_sort = np.argsort(invp)
            hard_idx = idx_sort[90:]
            neff_h = med([neff[i] for i in hard_idx])
            t10w_h = med([t10w[i] for i in hard_idx])
            rat_h = med([rat[i] for i in hard_idx])
            A(f"| {n} | {mlabel} | {med(invp):.2f} | {q90(invp):.2f} | {med(ent_n):.4f} | {med(neff):.1f} / {neff_h:.1f} | {med(t1w):.5f} | {med(t10w):.5f} / {t10w_h:.4f} | {med(t100w):.4f} | {med(rat):.2f}× / {rat_h:.1f}× |")
    A("")

A("### 1.3 Easy-10 vs. Median-10 vs. Hard-10 (B90 Tail) at n = 20 (`N_r0` vs. `Q`)")
A("")
A("Sorting the 100 test instances at n = 20 by each method's `1/p` (`Easy10` = ranks 0..9, `Med10` = ranks 45..54, `Hard10` = ranks 90..99):")
A("")
A("| Dataset | Method | Slice (10 inst) | Med 1/p | Med p_sol | Med N_eff | Med P_top10_wrong | Med Ratio P_top10_w / p_sol | Mean E_top1_wrong | Mean PE_z(top1_w) | Mean d_H(top1_w, sol) | Mean pv bit acc |")
A("|---|---|---|---|---|---|---|---|---|---|---|---|")
for label, d in [("4-SAT", d4), ("8-SAT", d8)]:
    rows = d["20"]
    for cname, mlabel in [("Q", "QAOA L4 (`Q`)"), ("QC_r0", "Chain-Proj (`QC_r0`)"), ("N_r0", "Native (`N_r0`)")]:
        invp = np.array([r["circuits"][cname]["inv_p"] for r in rows])
        order = np.argsort(invp)
        for sname, idxs in [("Easy10", order[:10]), ("Med10", order[45:55]), ("Hard10 (B90)", order[90:])]:
            sub = [rows[i] for i in idxs]
            A(f"| {label} | {mlabel} | {sname} | {med([r['circuits'][cname]['inv_p'] for r in sub]):.1f} | {med([r['circuits'][cname]['p_sol'] for r in sub]):.6f} | {med([r['circuits'][cname]['neff'] for r in sub]):.1f} | {med([r['circuits'][cname]['p_top10_wrong'] for r in sub]):.4f} | {med([r['circuits'][cname]['p_top10_wrong']/max(r['circuits'][cname]['p_sol'],1e-12) for r in sub]):.1f}× | {mean([r['circuits'][cname]['top1_E'] for r in sub]):.2f} | {mean([r['circuits'][cname]['top1_PE_z'] for r in sub]):.2f} | {mean([r['circuits'][cname]['top1_ham_sol'] for r in sub]):.2f} | {mean([r['pv_acc']*100 for r in sub]):.1f}% |")
A("")

A("---")
A("")
A("## 2. True Energy vs. Projected Energy of Wrong States (\"Ghost Minima\")")
A("")
A("A **ghost minimum** is an unsatisfying bitstring (`E(z) > 0`) whose 1D chain-projected energy is **strictly lower** than the minimum projected energy among all true satisfying assignments (`min_{z*: E(z*)=0} Proj(E)(z*)`).")
A("")
A("| Dataset | n | VarKept (block / min-cut r=0) | Corr(Proj(E), E) (r=0) | argmin Proj(E) is True Sol | Mean True E at argmin Proj(E) | Med / Mean Ghost Minima `N_ghost` (Uniform `Proj_unif(E)`) | Med Ghost Minima (`Proj_pv(E)`) | Med Ghost Minima (`ckpt_110` Layer-4 Effective Cost) |")
A("|---|---|---|---|---|---|---|---|---|")
for label, d in [("4-SAT", d4), ("8-SAT", d8)]:
    for n_str in ["8", "12", "16", "20"]:
        rows = d[n_str]
        vk_b = mean([r["proj_block"]["var_kept"] for r in rows])
        vk_r0 = mean([r["proj_r"][0]["var_kept"] for r in rows])
        cr = mean([r["proj_r"][0]["corr_PE_E"] for r in rows])
        as_sol = mean([r["proj_r"][0]["argmin_is_sol"] for r in rows]) * 100
        e_am = mean([r["proj_r"][0]["E_at_argmin"] for r in rows])
        ngu_med = med([r["proj_r"][0]["n_ghost_unif"] for r in rows])
        ngu_mean = mean([r["proj_r"][0]["n_ghost_unif"] for r in rows])
        ngpv_med = med([r["proj_r"][0]["n_ghost_pv"] for r in rows])
        ngeff_med = med([r["proj_r"][0]["n_ghost_eff"] for r in rows])
        A(f"| {label} | {n_str} | {vk_b:.4f} / {vk_r0:.4f} | {cr:.4f} | {as_sol:.1f}% | {e_am:.2f} | {ngu_med:.1f} / {ngu_mean:.1f} | {ngpv_med:.1f} | {ngeff_med:.1f} |")
A("")

A("### 2.2 Energy Profile and Energy-Shell Distribution of the Top Wrong States")
A("")
A("| Dataset | n | Method | Mean True E (top1 / top10w) | Mean Proj Energy z-score `PE_z(top1)` | % Instances Where `top1_wrong` Beats True Sol in `Proj_unif(E)` / `Proj_pv(E)` | Probability Mass by True Energy Shell: `E=0` / `E=1` / `E=2` / `E=3..4` / `E≥5` |")
A("|---|---|---|---|---|---|---|")
for label, d in [("4-SAT", d4), ("8-SAT", d8)]:
    for n_str in ["8", "12", "16", "20"]:
        rows = d[n_str]
        for cname, mlabel in [("Q", "QAOA L4 (`Q`)"), ("QC_r0", "Chain-Proj (`QC_r0`)"), ("N_pv_r0", "Native `Proj_pv` (`N_pv_r0`)"), ("N_r0", "Native `ckpt_110` (`N_r0`)")]:
            e1 = mean([r["circuits"][cname]["top1_E"] for r in rows])
            e10 = mean([r["circuits"][cname]["top10_wmean_E"] for r in rows])
            pez = mean([r["circuits"][cname]["top1_PE_z"] for r in rows])
            bu = mean([r["circuits"][cname]["top1_beats_sol_unif"] for r in rows]) * 100
            bpv = mean([r["circuits"][cname]["top1_beats_sol_pv"] for r in rows]) * 100
            esh = np.mean([r["circuits"][cname]["e_shells"] for r in rows], axis=0)
            e34 = esh[3] + esh[4]
            A(f"| {label} | {n_str} | {mlabel} | {e1:.2f} / {e10:.2f} | {pez:.2f} σ | {bu:.0f}% / {bpv:.0f}% | {esh[0]:.3f} / {esh[1]:.3f} / {esh[2]:.3f} / {e34:.3f} / {esh[5]:.3f} |")
A("")

A("---")
A("")
A("## 3. Hamming Distance Structure & GNN Cue Accuracy (`pv`, `E_loc`, Grover Mark)")
A("")
A("Where do the top wrong states sit in Hamming space relative to (a) the nearest true satisfying assignment `z*`, (b) the GNN's bit-marginal sign prediction `x_pv = (pv > 0.5)`, (c) the cavity-wise local optimistic minimum `x_E1 = argmin E_loc`, and (d) the global minimum of the projected Hamiltonian `argmin Proj_unif(E)`?")
A("")
A("| Dataset | n | GNN `pv` Bit Acc (`d_H(pv, sol)`) | `E(x_pv)` | Local Min `d_H(x_E1, sol)` (`E(x_E1)`) | % True-Sol Cavities in Grover Mark | Method | Mean `d_H(top1_w, sol)` | Wrong Cavities (of C) | Wrong Bits per Wrong Cavity | Mean `d_H(top1_w, pv / E1 / argmin_PE)` |")
A("|---|---|---|---|---|---|---|---|---|---|---|")
for label, d in [("4-SAT", d4), ("8-SAT", d8)]:
    for n_str in ["8", "12", "16", "20"]:
        rows = d[n_str]
        pv_acc = mean([r["pv_acc"] for r in rows]) * 100
        hpv_s = mean([r["ham_pv_sol"] for r in rows])
        epv = mean([r["E_at_pv"] for r in rows])
        he1_s = mean([r["ham_E1_sol"][0] for r in rows])
        ee1 = mean([r["E_at_E1"][0] for r in rows])
        smk = mean([r["sol_in_mark"][0] for r in rows]) * 100
        for cname, mlabel in [("Q", "QAOA (`Q`)"), ("QC_r0", "Chain (`QC_r0`)"), ("N_pv_r0", "`N_pv_r0`"), ("N_r0", "`N_r0`")]:
            hsol = mean([r["circuits"][cname]["top1_ham_sol"] for r in rows])
            cw = mean([r["circuits"][cname]["top1_cav_wrong"] for r in rows])
            bpw = mean([r["circuits"][cname]["top1_bpw_cav"] for r in rows])
            hpv = mean([r["circuits"][cname]["top1_ham_pv"] for r in rows])
            he1 = mean([r["circuits"][cname]["top1_ham_E1"] for r in rows])
            ham = mean([r["circuits"][cname]["top1_ham_argmin_PE"] for r in rows])
            A(f"| {label} | {n_str} | {pv_acc:.1f}% ({hpv_s:.2f}) | {epv:.2f} | {he1_s:.2f} ({ee1:.2f}) | {smk:.1f}% | {mlabel} | {hsol:.2f} | {cw:.2f} / {int(n_str)//4} | {bpw:.2f} / 4 | {hpv:.2f} / {he1:.2f} / {ham:.2f} |")
A("")

A("---")
A("")
A("## 4. Layer-by-Layer Trajectory (l = 1..4) and Component Ablations of `ckpt_110`")
A("")
A("### 4.1 Layer-by-Layer Evolution at n = 16 and n = 20")
A("")
A("| Dataset | n | Method | Med `p_sol` (l = 1 → 2 → 3 → 4) | Med `N_eff` (l = 1 → 2 → 3 → 4) | Mean True `<E>` (l = 1 → 2 → 3 → 4) | Mean `<Proj(E)_z>` (l = 1 → 2 → 3 → 4) | Med `P_top10_wrong` (l = 1 → 2 → 3 → 4) |")
A("|---|---|---|---|---|---|---|---|")
for label, d in [("4-SAT", d4), ("8-SAT", d8)]:
    for n_str in ["16", "20"]:
        rows = d[n_str]
        for cname, mlabel in [("Q", "QAOA L4 (`Q`)"), ("QC_r0", "Chain-Proj (`QC_r0`)"), ("N_pv_r0", "Native `Proj_pv` (`N_pv_r0`)"), ("N_r0", "Native `ckpt_110` (`N_r0`)")]:
            ps = np.median([r["circuits"][cname]["p_sol_l"] for r in rows], axis=0)
            nf = np.median([r["circuits"][cname]["neff_l"] for r in rows], axis=0)
            me = np.mean([r["circuits"][cname]["mE_l"] for r in rows], axis=0)
            mp = np.mean([r["circuits"][cname]["mPE_l"] for r in rows], axis=0)
            t10 = np.median([r["circuits"][cname]["top10_l"] for r in rows], axis=0)
            A(f"| {label} | {n_str} | {mlabel} | {ps[0]:.5f} → {ps[1]:.5f} → {ps[2]:.5f} → {ps[3]:.5f} | {nf[0]:.0f} → {nf[1]:.0f} → {nf[2]:.0f} → {nf[3]:.0f} | {me[0]:.2f} → {me[1]:.2f} → {me[2]:.2f} → {me[3]:.2f} | {mp[0]:.2f} → {mp[1]:.2f} → {mp[2]:.2f} → {mp[3]:.2f} | {t10[0]:.4f} → {t10[1]:.4f} → {t10[2]:.4f} → {t10[3]:.4f} |")
A("")

A("### 4.2 Component Ablation of `ckpt_110` (Placement r = 0, Med 1/p and B90 Q90 1/p)")
A("")
A("| Dataset | Variant | Description | n = 8 (Med / B90) | n = 12 (Med / B90) | n = 16 (Med / B90) | n = 20 (Med / B90) | n = 20 Med `N_eff` | n = 20 Med `P_top10_w` |")
A("|---|---|---|---|---|---|---|---|---|")
for label, d in [("4-SAT", d4), ("8-SAT", d8)]:
    for cname, vdesc in [
        ("Q", "All-to-all QAOA L4 (exact H_C)"),
        ("G", "All-to-all Global Cost L4 (GNN weights)"),
        ("QC_block", "Chain-Projected QAOA L4 (block order, no NN)"),
        ("QC_r0", "Chain-Projected QAOA L4 (min-cut r=0, no NN)"),
        ("N_unif_w", "`Proj_unif(E_w)` only (`pv=0.5, dS=0, dB=0`)"),
        ("N_unif_dSdB", "`Proj_unif(E_w) + dS + dB` (`pv=0.5`)"),
        ("N_pv_r0", "`Proj_pv(E_w)` only (`dS=0, dB=0`)"),
        ("N_grover_r0", "`Proj_pv(E_w) + grover(dS)` (`corr=0`)"),
        ("N_corr_r0", "`Proj_pv(E_w) + corr(dS, dB)` (`grover=0`)"),
        ("N_r0", "Full `ckpt_110` (`Proj_pv(E_w) + corr + grover`)"),
    ]:
        cells = []
        for n_str in ["8", "12", "16", "20"]:
            invp = [r["circuits"][cname]["inv_p"] for r in d[n_str]]
            cells.append(f"{med(invp):.2f} / {q90(invp):.2f}")
        nf20 = med([r["circuits"][cname]["neff"] for r in d["20"]])
        t10_20 = med([r["circuits"][cname]["p_top10_wrong"] for r in d["20"]])
        A(f"| {label} | `{cname}` | {vdesc} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {nf20:.1f} | {t10_20:.4f} |")
A("")

A("---")
A("")
A("## 5. Across R = 4 Placements: Do Different Variable-to-Cavity Permutations Lock into the Same or Different Wrong States?")
A("")
A("Each formula is evaluated across R = 4 variable-to-cavity placements (`r = 0` spectral min-cut + `r = 1, 2, 3` seeded random permutations), with top-10 wrong bitstrings mapped back to the **canonical variable ordering** so they can be compared directly across placements:")
A("")
A("| Dataset | n | Family | Med / B90 1/p (`r=0` min-cut) | Med / B90 1/p (`R=4` Round-Robin) | Med / B90 1/p (Best-of-4 `min_r 1/p_r`) | Med Spread `max_r p_r / min_r p_r` | Mean Distinct Top-1 Wrong States (of 4) | Mean Pairwise `d_H(top1_r1, top1_r2)` | Mean Pairwise Top-10 Overlap | Mean Union Size of Top-10 Sets (of 40) |")
A("|---|---|---|---|---|---|---|---|---|---|---|")
for label, d in [("4-SAT", d4), ("8-SAT", d8)]:
    for n_str in ["8", "12", "16", "20"]:
        rows = d[n_str]
        for fam, flabel in [("QC", "Chain-Proj QAOA (`QC`)"), ("N_pv", "Native `Proj_pv` (`N_pv`)"), ("N_full", "Native `ckpt_110` (`N_full`)")]:
            ip_r0 = [1.0 / max(r["placement_comp"][fam]["p_r"][0], 1e-12) for r in rows]
            ip_rr = [r["placement_comp"][fam]["inv_p_rr"] for r in rows]
            ip_best = [1.0 / max(r["placement_comp"][fam]["p_max"], 1e-12) for r in rows]
            sp = med([r["placement_comp"][fam]["spread"] for r in rows])
            nd = mean([r["placement_comp"][fam]["n_distinct_top1"] for r in rows])
            pwh = mean([r["placement_comp"][fam]["mean_pw_ham_top1"] for r in rows])
            ovl = mean([r["placement_comp"][fam]["mean_pw_top10_overlap"] for r in rows]) * 100
            usz = mean([r["placement_comp"][fam]["top10_union_size"] for r in rows])
            A(f"| {label} | {n_str} | {flabel} | {med(ip_r0):.2f} / {q90(ip_r0):.2f} | {med(ip_rr):.2f} / {q90(ip_rr):.2f} | {med(ip_best):.2f} / {q90(ip_best):.2f} | {sp:.2f}× | {nd:.2f} / 4 | {pwh:.2f} | {ovl:.1f}% | {usz:.1f} / 40 |")
A("")

A("### Key Takeaways from the R = 4 Placement Analysis")
A("1. **In Pure Chain-Projected QAOA (`QC`) and `N_pv`, different placements lock into completely different ghost minima:** At n = 20, the 4 placements produce `3.97 / 4` distinct top-1 wrong states (`pairwise d_H = 6.68` on 4-SAT, `7.47` on 8-SAT), with only `1.2%` (`4-SAT`) and `0.7%` (`8-SAT`) overlap between their top-10 wrong-state sets (`39.3 / 40` and `39.6 / 40` distinct states in the union). Because each permutation cuts a different subset of clauses, its ghost minima sit in different parts of the hypercube.")
A("2. **In Native `ckpt_110` on 4-SAT, the placement-invariant GNN bias `pv` pulls all 4 placements toward the SAME wrong cluster on hard instances:** Because `pv` depends only on the variable-clause graph (not the cavity permutation), `ckpt_110`'s top-1 wrong states across the 4 placements have **10× higher top-10 overlap** at n = 20 (`11.7%` vs `1.2%` for `QC`, with pairwise `d_H = 3.96` vs `6.68`). When `pv` is right (easy/median formulas), all 4 placements reinforce the true basin; when `pv` is wrong (the B90 tail), all 4 placements get dragged toward the same `x_pv` false attractor, explaining why `R=4` round-robin cannot rescue the B90 tail on 4-SAT (`B90 = 9,013` for `N_full` vs `6,030` for `N_pv` at n = 20).")

out_file = os.path.join(ROOT, "results/NATIVE_FAILURE_DIAGNOSIS.md")
with open(out_file, "w") as f:
    f.write("\n".join(lines) + "\n")
print("Wrote", out_file, "lines:", len(lines))

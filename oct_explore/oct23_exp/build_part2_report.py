import os, glob, json
import numpy as np

RES_DIR = "/workspace/qumode_sep/oct_explore/results/oct23_exp_part2"
REPORT_PATH = "/workspace/qumode_sep/oct_explore/results/OCT23_PART2_MULTICAV.md"

files = sorted(glob.glob(os.path.join(RES_DIR, "*.json")))
runs = []
for f in files:
    try:
        d = json.load(open(f))
        if "test" in d and "scaling" in d:
            runs.append(d)
    except Exception as e:
        print(f"Skipping {f}: {e}")

gray_shared = [r for r in runs if (r["args"].get("mixer") == "gray" or str(r["args"].get("tag","")).startswith("p2_gray_")) and r["args"].get("mode","shared") != "gnn"]
sd_shared   = [r for r in runs if r["args"].get("mode","shared") == "shared" and r["args"].get("mixer","sd") == "sd" and not str(r["args"].get("tag","")).startswith("p2_gray_")]
gray_gnn    = [r for r in runs if r["args"].get("mode") == "gnn" and r["args"].get("mixer") == "gray"]
sd_gnn      = [r for r in runs if r["args"].get("mode") == "gnn" and r["args"].get("mixer", "sd") == "sd"]

gray_shared.sort(key=lambda r: (r["args"]["L"], r["args"].get("ntrain8", 2000)))
sd_shared.sort(key=lambda r: (r["args"]["L"], r["args"]["pairs"], r["args"].get("ntrain8", 2000)))
gray_gnn.sort(key=lambda r: (r["args"]["L"], r["args"].get("arch", ""), int(r["args"].get("clause_w", 0))))
sd_gnn.sort(key=lambda r: (r["args"]["L"], r["args"]["pairs"], r["args"].get("arch", ""), int(r["args"].get("clause_w", 0))))

lines = []
lines.append("# Oct 23 Exploration — Part 2: Multi-Cavity Scaling (`K = 4` Bits/Cavity, `n = 8, 12, 16, 20`)")
lines.append("")
lines.append("## 0. Experimental Protocol & Audit Compliance")
lines.append("- **Encoding**: `K = 4` bits per cavity (`D = 2^4 = 16` computational Fock states `0..15`, `C = n / 4 in {2, 3, 4, 5}` cavities for `n in {8, 12, 16, 20}`, cutoff `N_Fock = 40`).")
lines.append("- **Strict Computational-Subspace SNAP**: All SNAP gates (both in `k_prep = 2, 3` state preparation and in all `L` circuit layers) act **exclusively on computational Fock states `0..15`**, with strictly `0` phase on leaked Fock levels `f >= 16`.")
lines.append("- **Leakage Accounted as Failure**: Any Fock population outside `0..15` in any cavity is excluded from `p_succ` and reported explicitly as `Leak%`.")
lines.append("- **No Classical Pre-Solving (`relabel = none`)**: Canonical variable ordering and polarity (`flips = 0`), with raw factor-graph degree features only (`fv[:, :, 3:7] = 0`). The vacuum `|0,0,...,0>` pre-solve rate is `1%` at `n = 8` and `0%` at `n = 12, 16, 20`.")
lines.append("- **Size-Transferable Training**: Shared parameters and GNN weights are trained **strictly on `n = 8, 12` training instances** (`TRB = 51e9`, selected on `n = 12` validation `VAB = 53e9`) and evaluated **zero-shot at `n = 16, 20`** on 100 unseen test instances per `n` (`TEB = 52e9`).")
lines.append("")
lines.append("### Classical Baselines on the 100 Unseen Test Formulas (`TEB = 52e9`)")
lines.append("| Classical Baseline / Reference | `n = 8` (Med / B90) | `n = 12` (Med / B90) | `n = 16` (Med / B90) | `n = 20` (Med / B90) | Growth `b^n` (Med / B90) |")
lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
lines.append("| **WalkSATlm Tries** (`p = 0.30`, 500 runs/inst) | `2.01 / 3.74` | `2.92 / 6.28` | `4.55 / 9.09` | `5.63 / 14.2` | **`1.092^n / 1.116^n`** |")
lines.append("| **WalkSATlm Clause Lookups** (`p = 0.30`, 500 runs/inst) | `13.99 / 29.40` | `32.5 / 76.1` | `71.0 / 147` | `110 / 290` | **`1.191^n / 1.207^n`** |")
lines.append("| **Uniform Random Guessing (`2^n` / Solutions)** | `42.67 / 128.00` | `585.14 / 2048.00` | `9362.29 / 32768.00` | `149796.6 / 524288.0` | **`1.970^n / 2.000^n`** |")
lines.append("")

def fmt_cell(m, n):
    s = f"{m['median_inv_p']:.2f} / {m['B90_inv_p']:.2f}"
    if m.get("beats_ws_tries_med") and m.get("beats_ws_tries_b90"):
        return f"**{s}** (★T)"
    elif m.get("beats_ws_look_med") and m.get("beats_ws_look_b90"):
        return f"**{s}** (✓L)"
    elif m.get("beats_ws_look_med"):
        return f"**{s}**"
    return s

if gray_shared:
    lines.append("## 1A-i. Stage-1 Ideal Cavity Gray Mixer `U_gray(β_l)` + Global Cost Only (`2L` shared params `(β_l, γ_l)`; no free SNAP θ)")
    lines.append("")
    lines.append("> **Stage-1 shared (`2L` parameters):** exact computational-subspace `U_gray(β_l)` + cost phase `γ_l` with **no** free per-layer SNAP phases (`θ = 0`). State preparation uses `k_prep = 2` physical `0..15` `SNAP+D` blocks (`n=20` state-prep leakage `1 - (1 - 0.002887)^5 = 1.44%`). ★T = dual-beats WalkSATlm tries (Med **and** B90); ✓L = dual-beats WalkSATlm lookups. `p2_gray_L15` at `n=20` is **not** ★T (`B90 = 14.52 ≰ 14.2`).")
    lines.append("")
    lines.append("| Tag | Mode | `L` | `N_tr(8,12)` | `n = 8` (Med / B90) | `n = 12` (Med / B90) | `n = 16` (Med / B90) | `n = 20` (Med / B90) | Growth `b^n` (Med / B90) | Zero-Oracle `n=12 / n=20` | `n=20` Leak% |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in gray_shared:
        a = r["args"]; te = r["test"]; sc = r["scaling"]
        ntr = f"{a.get('ntrain8',2000)},{a.get('ntrain12',500)}"
        z_str = f"`{te['12']['zero_oracle_med']:.0f}` / `{te['20']['zero_oracle_med']:.0f}`"
        lk20 = f"{te['20']['mean_leak']*100:.2f}%"
        lines.append(f"| `{a['tag']}` | `shared` | {a['L']} | `{ntr}` | {fmt_cell(te['8'],8)} | {fmt_cell(te['12'],12)} | {fmt_cell(te['16'],16)} | {fmt_cell(te['20'],20)} | **`{sc['b_median']:.4f}^n / {sc['b_B90']:.4f}^n`** | {z_str} | {lk20} |")
    lines.append("")

if gray_gnn:
    lines.append("## 1A-ii. Ideal Cavity Gray Mixer `U_gray(β_{c,l})` + Per-Cavity `0..15` SNAP Phases + Factor-Graph GNN")
    lines.append("")
    lines.append("> **Gray + `0..15` SNAP + GNN (`relabel = none`):** per cavity `c` and layer `l`, the GNN outputs `β_{c,l}` and **two** computational-subspace `0..15` diagonal SNAP gates (`Vp = exp(-i sout_{c,l}) · U_gray(β_{c,l}) · exp(-i θ_{c,l})`), plus optional per-clause cost weights `w_b` (`clause_w`). State prep uses the same `k_prep = 2` physical `0..15` `SNAP+D` blocks (`n=20` leak `1.44%`).")
    lines.append(">")
    lines.append("> **Finite-size ★T vs asymptotic B90 slope (verified by `@qumode_checker`):** `p2_gnn_gray_L15_mp64_cw` achieves ★T dual-beat vs WalkSATlm tries at every tested size `n = 8, 12, 16, 20` (**1.32/2.27, 1.90/4.10, 2.76/7.88, 3.95/12.18** vs **2.01/3.74, 2.92/6.28, 4.55/9.09, 5.63/14.2**), with median growth **1.0957ⁿ** matching WalkSATlm tries (**1.092ⁿ**). However, its B90 growth is **1.1530ⁿ** vs WalkSATlm tries B90 **1.116ⁿ** (and bare `p2_gray_L15` **1.1376ⁿ**): the GNN wins `n ≤ 20` via a ~36% lower B90 prefactor (`C_B90 = 0.72` vs `1.12`), not a flatter B90 exponent (extrapolating the fit crosses WalkSATlm tries B90 just above `n ≈ 24`).")
    lines.append("")
    lines.append("| Tag | Arch (`H`) | `L` | `N_tr(8,12)` | `cw` | `n = 8` (Med / B90) | `n = 12` (Med / B90) | `n = 16` (Med / B90) | `n = 20` (Med / B90) | Growth `b^n` (Med / B90) | Zero-Oracle `n=12 / n=20` | `n=20` Leak% |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in gray_gnn:
        a = r["args"]; te = r["test"]; sc = r["scaling"]
        arch = f"{a.get('arch', 'mp')} (H={a.get('H',64)}, r={a.get('rounds',3)})"
        ntr = f"{a.get('ntrain8',2000)},{a.get('ntrain12',500)}"
        cw = "yes" if a.get("clause_w") else "no"
        z_str = f"`{te['12']['zero_oracle_med']:.0f}` / `{te['20']['zero_oracle_med']:.0f}`"
        lk20 = f"{te['20']['mean_leak']*100:.2f}%"
        lines.append(f"| `{a['tag']}` | `{arch}` | {a['L']} | `{ntr}` | {cw} | {fmt_cell(te['8'],8)} | {fmt_cell(te['12'],12)} | {fmt_cell(te['16'],16)} | {fmt_cell(te['20'],20)} | **`{sc['b_median']:.4f}^n / {sc['b_B90']:.4f}^n`** | {z_str} | {lk20} |")
    lines.append("")

lines.append("## 1B. Direct Physical `k = 1, 2, 3` `SNAP+D` Blocks per Layer — Shared Parameters (L-BFGS-B on `n = 8 + 12`)")
lines.append("| Tag | `L` | `k` (`SNAP+D`/layer) | Blocks/Cav (`k_p + L*k`) | `N_tr(8,12)` | `n = 8` (Med / B90) | `n = 12` (Med / B90) | `n = 16` (Med / B90) | `n = 20` (Med / B90) | Growth `b^n` (Med / B90) | `n=12` Zero-Oracle | `n=20` Leak% |")
lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for r in sd_shared:
    a = r["args"]; te = r["test"]; sc = r["scaling"]
    L = a["L"]; k = a["pairs"]; blk = a["kprep"] + L * k
    ntr = f"{a.get('ntrain8',2000)},{a.get('ntrain12',500)}"
    z12 = f"{te['12']['zero_oracle_med']:.0f}/{te['12']['zero_oracle_B90']:.0f}"
    lk20 = f"{te['20']['mean_leak']*100:.2f}%"
    lines.append(f"| `{a['tag']}` | {L} | {k} | {blk} | `{ntr}` | {fmt_cell(te['8'],8)} | {fmt_cell(te['12'],12)} | {fmt_cell(te['16'],16)} | {fmt_cell(te['20'],20)} | **`{sc['b_median']:.4f}^n / {sc['b_B90']:.4f}^n`** | `{z12}` | {lk20} |")
lines.append("")

lines.append("## 2. Direct Physical `SNAP+D` Factor-Graph GNN Scaling (`relabel = none`, Trained on `n = 8, 12` -> Zero-Shot `n = 16, 20`)")
lines.append("")
lines.append("> **Median & B90 vs `p2_shared_L15_k3`:** While `k = 2` GNNs (`L=12, 15`) beat `p2_shared_L15_k3` on medians at all `n` (`62.45` at `n=20`, **1.2573ⁿ**) while trailing on B90 at `n = 12, 16, 20`, stepping up to **`L = 15, k = 3`** (`p2_gnn_L15_k3_mp64_nocw`, `p2_gnn_L15_k3_mp64_cw`, and `p2_gnn_L15_k3_mp128_cw`) fixes the tail completely: all three `L = 15, k = 3` GNNs **dual-beat `p2_shared_L15_k3` (and `ntr10000`) on BOTH Median and B90 at every `n = 8, 12, 16, 20`** (`p2_gnn_L15_k3_mp64_nocw` without `clause_w`: **3.17/10.86, 7.40/38.29, 18.60/124.33, 42.88/185.02**, growth **1.2441ⁿ / 1.2739ⁿ**, leak `7.57%`; `p2_gnn_L15_k3_mp64_cw`: **3.42/9.73, 8.00/34.98, 20.33/92.82, 43.91/181.65**, growth **1.2395ⁿ / 1.2762ⁿ**, leak `8.00%`), and **all three achieve dual WalkSATlm lookup beats (✓L) across all `n = 8, 12, 16, 20`**.")
lines.append("")
lines.append("| Tag | `L` | `k` | Arch (`H`) | `clause_w` | `n = 8` (Med / B90) | `n = 12` (Med / B90) | `n = 16` (Med / B90) | `n = 20` (Med / B90) | Growth `b^n` (Med / B90) | `n=12` Zero-Oracle | `n=20` Leak% |")
lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for r in sd_gnn:
    a = r["args"]; te = r["test"]; sc = r["scaling"]
    L = a["L"]; k = a["pairs"]
    arch = f"{a['arch']} (H={a['H']}, r={a['rounds']})"
    cw = "yes" if a.get("clause_w") else "no"
    z12 = f"{te['12']['zero_oracle_med']:.0f}/{te['12']['zero_oracle_B90']:.0f}"
    lk20 = f"{te['20']['mean_leak']*100:.2f}%"
    lines.append(f"| `{a['tag']}` | {L} | {k} | `{arch}` | {cw} | {fmt_cell(te['8'],8)} | {fmt_cell(te['12'],12)} | {fmt_cell(te['16'],16)} | {fmt_cell(te['20'],20)} | **`{sc['b_median']:.4f}^n / {sc['b_B90']:.4f}^n`** | `{z12}` | {lk20} |")
lines.append("")

with open(REPORT_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {REPORT_PATH} with {len(gray_shared)} gray_shared, {len(sd_shared)} sd_shared, {len(gray_gnn)} gray_gnn, {len(sd_gnn)} sd_gnn runs.")

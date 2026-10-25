import os, glob, json
import numpy as np

RES_DIR = "/workspace/qumode_sep/oct_explore/results/oct23_exp_part2"
REPORT_PATH = "/workspace/qumode_sep/oct_explore/results/OCT23_PART2_MULTICAV.md"

files = sorted(glob.glob(os.path.join(RES_DIR, "*.json")))
runs = []
for f in files:
    base = os.path.basename(f)
    if "INVALID" in base or "quarantine" in base.lower():
        print(f"Skipping quarantine {base}")
        continue
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
sd_shared.sort(key=lambda r: (r["args"]["L"], r["args"]["pairs"], r["args"].get("ntrain8", 2000), r["args"]["tag"]))
gray_gnn.sort(key=lambda r: (r["args"]["L"], r["args"].get("arch", ""), int(r["args"].get("clause_w", 0))))
sd_gnn.sort(key=lambda r: (r["args"]["L"], r["args"]["pairs"], r["args"].get("arch", ""), int(r["args"].get("clause_w", 0)), r["args"].get("ntrain8", 2000), r["args"]["tag"]))

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
    lines.append("> **Finite-size ★T vs asymptotic B90 slope (verified by `@qumode_checker`):** `p2_gnn_gray_L15_mp64_cw` achieves ★T dual-beat vs WalkSATlm tries at every tested size `n = 8, 12, 16, 20` (**1.32/2.27, 1.90/4.10, 2.76/7.88, 3.95/12.18** vs **2.01/3.74, 2.92/6.28, 4.55/9.09, 5.63/14.2**), with median growth **1.0957ⁿ** matching WalkSATlm tries (**1.092ⁿ**). However, its B90 growth is **1.1530ⁿ** vs WalkSATlm tries B90 **1.116ⁿ** (and bare `p2_gray_L15` **1.1376ⁿ**): the GNN wins `n ≤ 20` via a ~36% lower B90 prefactor (`C_B90 = 0.72` vs `1.12`), not a flatter B90 exponent (extrapolating the fit crosses WalkSATlm tries B90 just above `n ≈ 24`). WAVE4 `ds` arch `p2_gnn_gray_L15_ds_cw` (**1.37/2.38, 2.05/4.51, 3.11/9.68, 4.17/13.69**, **1.0985ⁿ / 1.1623ⁿ**) gets ★T at `n=8,12,20` but misses `n=16` B90 (`9.68 ≰ 9.09`) and trails `mp64_cw` on Med+B90 at every `n` (B90 slope also steeper than `mp64_cw`'s **1.1530ⁿ**).")
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

lines.append("## 1B. Direct Physical `k = 1, 2, 3, 4` `SNAP+D` Blocks per Layer — Shared Parameters (L-BFGS-B on `n = 8 + 12`)")
lines.append("")
lines.append("> **Trajectory-distilled Gray→physical init (`grayinit`, WAVE4):** replacing 4-start small-angle `N(0, 0.15..0.45)` `calibrate_single_layer` (`single-layer-calibrated shared`) with (a) 20 full-angle `U(-π, π)` SNAP multi-starts + bidirectional layer continuation (`val12 = 16.52/112.20` at `k=3`, `14.59/48.20` at `k=4`) and (b) Stage-2B many-body state-trajectory distillation matching `|ψ_gray⟩` on `n=8, 12` train instances (dropping pre-Stage-3 `val12` to `5.91/30.05` at `k=3` and `5.93/25.60` at `k=4`), followed by L-BFGS-B refinement, yields a large jump: `p2_shared_L15_k4_grayinit` (**2.56/6.46, 4.84/18.31, 9.37/36.48, 16.89/77.63**, **1.1712ⁿ / 1.2259ⁿ**, dual ✓L all `n`, `n=20` leak 6.97%) and `p2_shared_L15_k3_grayinit` (**2.78/7.35, 5.12/25.36, 10.88/50.86, 23.89/102.91**, **1.1974ⁿ / 1.2403ⁿ**, dual ✓L all `n`). Both dual-beat the best single-layer-calibrated shared run `p2_shared_L15_k3_ntr10000` (**4.88/12.55 … 93.75/304.42**, **1.2768ⁿ / 1.2971ⁿ**) on Med+B90 at every `n`, and for the first time give physical shared dual ✓L through `n=20`. Still no ★T vs WalkSATlm tries (closest shared at `n=8`: med 2.56 ≰ 2.01). Grayinit-seeded physical GNNs: **WAVE10 complete** — `…_mp64_cw_ntr10000_n12_2000_s2000` (**1.66/3.76…9.74/35.41**, **1.1603ⁿ / 1.2071ⁿ**) was then closest-to-★T (`n=8` B90 **3.76**, gap **0.02**). **WAVE11 complete:** `…_n12_2000_s3000` (**1.58/3.69, 2.86/10.58, 5.59/28.50, 11.00/39.25**, **1.1760ⁿ / 1.2240ⁿ**, dual ✓L all `n`) took first physical ★T at `n=8` with **best Med (`1.58`)**; sibling ★T `…_n12_3000_s1600` still holds **best physical `n=16` B90 (`19.73`)**. **WAVE12 complete:** `…_n12_3000_s2400` (**1.63/3.47, 2.68/9.85, 5.04/22.74, 9.15/34.57**, **1.1562ⁿ / 1.2133ⁿ**, dual ✓L all `n`) is **best physical ★T at `n=8`** (B90 **3.47 ≤ 3.74**) + **best physical Med/B90 at `n=12` (`2.68`/`9.85`)**; sibling ★T `…_n12_3000_s2000` (**1.65/3.59, 2.73/10.49, 5.06/22.44, 8.82/35.63**, **1.1517ⁿ / 1.2106ⁿ**) keeps **best physical Med at `n=20` (`8.82`)** + **flattest physical Med (`1.1517ⁿ`)**. WAVE12 misses: `…_n12_2000_s3600` (**1.62/3.96…**, dual ✓L only, no ★T — longer Adam on `n12_2000` overfit); `…_mp128_…_n12_2000_s2000` (**1.64/3.74…**) misses ★T (`B90 3.7445 ≰ 3.74`). Remaining gaps: `n=12` B90 (need ≤6.28; best **9.85**), `n=16`/`n=20` Med+B90 vs tries, flatter growth toward WS **1.092ⁿ/1.116ⁿ**. Prior `k3` grayinit GNN still dual-beats its parent `k3_grayinit` shared.")
lines.append("")
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
lines.append("> **Median & B90 vs shared (verified by `@qumode_checker`):** At `N_tr=2000`, `mp64_cw` (`92.82` at `n=16` B90) and `mp128_cw` (`81.59`) dual-beat both single-layer-calibrated shared runs (`p2_shared_L15_k3` and `p2_shared_L15_k3_ntr10000`, `n=16` B90 `127.98` and `114.84`) on Med+B90 at every `n = 8, 12, 16, 20`, while `mp64_nocw` (`N_tr=2000`, `n=16` B90 `124.33`) dual-beats `p2_shared_L15_k3` at every `n` and `p2_shared_L15_k3_ntr10000` at `n = 8, 12, 20` (`124.33 > 114.84` on `n=16` B90). WAVE4 `N_tr=5000` improves further: best single-layer-init physical GNN is `p2_gnn_L15_k3_mp64_cw_ntr5000` (**2.91/8.53, 6.23/31.92, 15.64/62.92, 31.46/128.97**, **1.2231ⁿ / 1.2469ⁿ**, dual ✓L all `n`), ahead of sibling `mp64_nocw_ntr5000` (**3.19/9.56 … 38.67/149.51**, **1.2335ⁿ / 1.2536ⁿ**). WAVE5 grayinit `…_k4_grayinit_mp64_cw_ntr5000` (**1.88/4.96, 3.49/11.34, 6.00/31.40, 11.23/43.13**, **1.1593ⁿ / 1.2064ⁿ**, dual ✓L all `n`) was first physical B90 growth matching WalkSATlm lookups (**1.207ⁿ**). WAVE6 true-`N_tr=10000` `…_mp64_cw_ntr10000` (**1.77/4.88, 3.11/12.89, 5.46/31.42, 10.61/46.59**, **1.1602ⁿ / 1.2111ⁿ**, dual ✓L all `n`) took Med at every `n` vs WAVE5. **WAVE7** held flattest B90: `…_mp64_cw_ntr10000_s1200` (**1.78/4.37 … 10.85/38.87**, **1.1631ⁿ / 1.2050ⁿ**). **WAVE10 complete:** `…_ntr10000_n12_2000_s2000` (**1.66/3.76…9.74/35.41**, **1.1603ⁿ / 1.2071ⁿ**) was then closest-to-★T. WAVE7 `…_mp64_cw_ntr10000_s1200` remains flattest B90 (**1.2050ⁿ**). **WAVE11 complete:** `…_ntr10000_n12_2000_s3000` (**1.58/3.69…**) first physical ★T + best Med at `n=8`; sibling ★T `…_n12_3000_s1600` still **best physical `n=16` B90 (`19.73`)**. **WAVE12 complete:** `…_n12_3000_s2400` (**1.63/3.47, 2.68/9.85, 5.04/22.74, 9.15/34.57**, **1.1562ⁿ / 1.2133ⁿ**) is **best physical ★T at `n=8`** (B90 **3.47**) + **best physical Med/B90 at `n=12`**; sibling ★T `…_n12_3000_s2000` (**1.65/3.59…8.82/35.63**, **1.1517ⁿ / 1.2106ⁿ**) keeps **best Med at `n=20`** + **flattest Med (`1.1517ⁿ`)**. WAVE12 misses: `…_n12_2000_s3600` (**1.62/3.96…**, dual ✓L only, no ★T — longer Adam on `n12_2000` overfit); `…_mp128_…_n12_2000_s2000` misses ★T (`3.7445 ≰ 3.74`). Gaps: `n=12` B90 ≤6.28 (best **9.85**), `n=16`/`n=20` tries, growth vs WS **1.092ⁿ/1.116ⁿ**. Clean layer-truncate `p2_gnn_L12_k3_mp64_cw` replaces quarantined `…_INVALID_L15flatinit.")
lines.append("")
lines.append("| Tag | `L` | `k` | Arch (`H`) | `N_tr(8,12)` | `clause_w` | `n = 8` (Med / B90) | `n = 12` (Med / B90) | `n = 16` (Med / B90) | `n = 20` (Med / B90) | Growth `b^n` (Med / B90) | `n=12` Zero-Oracle | `n=20` Leak% |")
lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for r in sd_gnn:
    a = r["args"]; te = r["test"]; sc = r["scaling"]
    L = a["L"]; k = a["pairs"]
    arch = f"{a['arch']} (H={a['H']}, r={a['rounds']})"
    ntr = f"{a.get('ntrain8',2000)},{a.get('ntrain12',500)}"
    cw = "yes" if a.get("clause_w") else "no"
    z12 = f"{te['12']['zero_oracle_med']:.0f}/{te['12']['zero_oracle_B90']:.0f}"
    lk20 = f"{te['20']['mean_leak']*100:.2f}%"
    lines.append(f"| `{a['tag']}` | {L} | {k} | `{arch}` | `{ntr}` | {cw} | {fmt_cell(te['8'],8)} | {fmt_cell(te['12'],12)} | {fmt_cell(te['16'],16)} | {fmt_cell(te['20'],20)} | **`{sc['b_median']:.4f}^n / {sc['b_B90']:.4f}^n`** | `{z12}` | {lk20} |")
lines.append("")

with open(REPORT_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {REPORT_PATH} with {len(gray_shared)} gray_shared, {len(sd_shared)} sd_shared, {len(gray_gnn)} gray_gnn, {len(sd_gnn)} sd_gnn runs.")

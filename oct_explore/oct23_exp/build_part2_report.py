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

gray_shared.sort(key=lambda r: r["args"]["L"])
sd_shared.sort(key=lambda r: (r["args"]["L"], r["args"]["pairs"], r["args"].get("ntrain8", 2000)))
gray_gnn.sort(key=lambda r: (r["args"]["L"], r["args"].get("arch", ""), int(r["args"].get("clause_w", 0))))
sd_gnn.sort(key=lambda r: (r["args"]["L"], r["args"]["pairs"], r["args"].get("arch", ""), int(r["args"].get("clause_w", 0))))

lines = []
lines.append("# Oct 23 Exploration — Part 2: Multi-Cavity Scaling (`K = 4` Bits/Cavity, `n = 8, 12, 16, 20`)")
lines.append("")
lines.append("## 0. Experimental Protocol & Audit Compliance")
lines.append("- **Encoding**: `K = 4` bits per cavity (`D = 2^4 = 16` computational Fock states `0..15`, `C = n / 4 in {2, 3, 4, 5}` cavities for `n in {8, 12, 16, 20}`, cutoff `N_Fock = 40`).")
lines.append("- **Strict Computational-Subspace SNAP**: All SNAP gates (both in `k_prep = 2` state preparation and in all `L` circuit layers) act **exclusively on computational Fock states `0..15`**, with strictly `0` phase on leaked Fock levels `f >= 16`.")
lines.append("- **Leakage Accounted as Failure**: Any Fock population outside `0..15` in any cavity is excluded from `p_succ` and reported explicitly as `Leak%`.")
lines.append("- **No Classical Pre-Solving (`relabel = none`)**: Canonical variable ordering and polarity (`flips = 0`), with raw factor-graph degree features only (`fv[:, :, 3:7] = 0`). The vacuum `|0,0,...,0>` pre-solve rate is `1%` at `n = 8` and `0%` at `n = 12, 16, 20`.")
lines.append("- **Size-Transferable Training**: Shared parameters and GNN weights are trained **strictly on `n = 8, 12` training instances** (`TRB = 51e9`, selected on `n = 12` validation `VAB = 53e9`) and evaluated **zero-shot at `n = 16, 20`** on 100 unseen test instances per `n` (`TEB = 52e9`).")
lines.append("")
lines.append("### Classical Baselines on the 100 Unseen Test Formulas (`TEB = 52e9`)")
lines.append("| Classical Baseline / Reference | `n = 8` (Med / B90) | `n = 12` (Med / B90) | `n = 16` (Med / B90) | `n = 20` (Med / B90) | Growth `b^n` (Med / B90) |")
lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
lines.append("| **WalkSATlm Tries** (`p = 0.30`, 500 runs/inst) | `2.01 / 3.74` | `2.92 / 6.28` | `4.55 / 9.09` | `5.63 / 14.2` | **`1.092^n / 1.116^n`** |")
lines.append("| **WalkSATlm Clause Lookups** (`p = 0.30`, 500 runs/inst) | `13.99 / 29.40` | `32.51 / 76.12` | `78.93 / 214.08` | `111.79 / 356.11` | **`1.191^n / 1.236^n`** |")
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

if gray_shared or gray_gnn:
    lines.append("## 1A. Stage-1 Ideal Cavity Gray Mixer `U_gray(β)` + Global Cost Only (`2L` params `(β_l, γ_l)`; no free SNAP θ)")
    lines.append("| Tag | Mode | Arch / Init | `L` | `cw` | `n = 8` (Med / B90) | `n = 12` (Med / B90) | `n = 16` (Med / B90) | `n = 20` (Med / B90) | Growth `b^n` (Med / B90) | Zero-Oracle `n=12 / n=20` |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in gray_shared + gray_gnn:
        a = r["args"]; te = r["test"]; sc = r["scaling"]
        arch = "Shared-2L" if a.get("mode") == "shared" else a.get("arch", "mp")
        cw = "yes" if a.get("clause_w") else "no"
        z_str = f"`{te['12']['zero_oracle_med']:.0f}` / `{te['20']['zero_oracle_med']:.0f}`"
        lines.append(f"| `{a['tag']}` | `{a.get('mode')}` | `{arch}` | {a['L']} | {cw} | {fmt_cell(te['8'],8)} | {fmt_cell(te['12'],12)} | {fmt_cell(te['16'],16)} | {fmt_cell(te['20'],20)} | **`{sc['b_median']:.4f}^n / {sc['b_B90']:.4f}^n`** | {z_str} |")
    lines.append("")

lines.append("## 1B. Direct Physical `k = 1, 2, 3` `SNAP+D` Blocks per Layer — Shared Parameters (L-BFGS-B on `n = 8 + 12`)")
lines.append("| Tag | `L` | `k` (`SNAP+D`/layer) | Blocks/Cav (`2 + L*k`) | `N_tr(8,12)` | `n = 8` (Med / B90) | `n = 12` (Med / B90) | `n = 16` (Med / B90) | `n = 20` (Med / B90) | Growth `b^n` (Med / B90) | `n=12` Zero-Oracle | `n=20` Leak% |")
lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for r in sd_shared:
    a = r["args"]; te = r["test"]; sc = r["scaling"]
    L = a["L"]; k = a["pairs"]; blk = a["kprep"] + L * k
    ntr = f"{a.get('ntrain8',2000)},{a.get('ntrain12',500)}"
    z12 = f"{te['12']['zero_oracle_med']:.0f}/{te['12']['zero_oracle_B90']:.0f}"
    lk20 = f"{te['20']['mean_leak']*100:.2f}%"
    lines.append(f"| `{a['tag']}` | {L} | {k} | {blk} | `{ntr}` | {fmt_cell(te['8'],8)} | {fmt_cell(te['12'],12)} | {fmt_cell(te['16'],16)} | {fmt_cell(te['20'],20)} | **`{sc['b_median']:.4f}^n / {sc['b_B90']:.4f}^n`** | `{z12}` | {lk20} |")
lines.append("")

lines.append("## 2. Pure Factor-Graph GNN Scaling (`relabel = none`, Trained on `n = 8, 12` -> Zero-Shot `n = 16, 20`)")
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

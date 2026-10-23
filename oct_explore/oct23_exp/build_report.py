"""Generate /workspace/qumode_sep/oct_explore/results/OCT23_PART1_2CAV.md from finished JSONs."""
import os, json, glob, time
import numpy as np

RES_DIR = "/workspace/qumode_sep/oct_explore/results/oct23_exp"
OUT_MD = "/workspace/qumode_sep/oct_explore/results/OCT23_PART1_2CAV.md"

def load_all():
    out = {}
    for p in sorted(glob.glob(os.path.join(RES_DIR, "*.json"))):
        name = os.path.basename(p)[:-5]
        if name == "preps": continue
        try:
            out[name] = json.load(open(p))
        except Exception:
            pass
    return out

def fmt_row(tag, J):
    a = J.get("args", {})
    te = J.get("test", {})
    va = J.get("val", {})
    tr = J.get("train", {})
    nf_chk_key = [k for k in J.keys() if k.startswith("test_NF")]
    chk = J[nf_chk_key[0]] if nf_chk_key else te
    med = te.get("median_inv_p", 0.0)
    b90 = te.get("B90_inv_p", 0.0)
    c_med = chk.get("median_inv_p", 0.0)
    c_b90 = chk.get("B90_inv_p", 0.0)
    lk = chk.get("mean_leak", 0.0)
    v_med = va.get("median_inv_p", 0.0)
    v_b90 = va.get("B90_inv_p", 0.0)
    extra = ""
    if "rr4_med" in te:
        extra = f" (RR4: {te['rr4_med']:.2f}/{te['rr4_B90']:.2f}, best4: {te['best4_med']:.2f}/{te['best4_B90']:.2f})"
    elif "best_of_4_med" in te:
        extra = f" (r0: {te['r0_med']:.2f}/{te['r0_B90']:.2f}, best4: {te['best_of_4_med']:.2f}/{te['best_of_4_B90']:.2f})"
    return f"| `{tag}` | {a.get('n')} | `{a.get('sched')}` (`{a.get('order')}`) | `{a.get('gate')}` | {a.get('ntrain')} | `{a.get('relabel')}` | {v_med:.2f} / {v_b90:.2f} | **{med:.2f} / {b90:.2f}**{extra} | {c_med:.2f} / {c_b90:.2f} | {lk*100:.2f}% |"

def main():
    runs = load_all()
    preps = json.load(open(os.path.join(RES_DIR, "preps.json"))) if os.path.exists(os.path.join(RES_DIR, "preps.json")) else {}
    lines = [
        "# Oct 23 Part 1: Two-Cavity Experimental Circuit (`n = 6` and `n = 8`, <= 5 Blocks Total)",
        "",
        f"*Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')} (UTC+8). All SNAP gates strictly restricted to computational Fock levels `0..D-1` (`0` phase on leaked levels `f >= D`); all Fock leakage (`f >= D` or ancilla in `|e>`) counted as failure and verified at `N_Fock = 40` (`n=6`) and `N_Fock = 56` (`n=8`).*",
        "",
        "## 1. Classical WalkSATlm Baselines on the 100 Unseen Test Formulas (`TEB = 52e9`)",
        "",
        "| Problem Size `n` | Layout | WalkSATlm Tries (Med / B90) | WalkSATlm Lookups (Med / B90) | Random Guess `2^n / n_sol` (Med / B90) |",
        "|---|---|---|---|---|",
        "| **n = 6** | 2 cavities x 3 bits (`D = 8`) | **1.691 / 2.816** | **8.466 / 16.039** | 64.0 / 64.0 |",
        "| **n = 8** | 2 cavities x 4 bits (`D = 16`) | **2.014 / 3.739** | **13.989 / 29.396** | 128.0 / 256.0 |",
        "",
        "## 2. State Preparation Fidelity from Vacuum (`0..D-1` Computational SNAP Phases Only)",
        "",
        "| Gate & Cavity Size | `k = 1` Block `F` (leak) | `k = 2` Blocks `F` (leak) | `k = 3` Blocks `F` (leak) |",
        "|---|---|---|---|",
    ]
    for K, D in [(2, 4), (3, 8), (4, 16)]:
        for gname, label in [("sd", f"SNAP+D (`K={K}, D={D}`)"), ("ecd", f"ECD ancilla (`K={K}, D={D}`)"), ("ecdt", f"ECD hybrid (`1t+{K}c, 2D={2*D}`)")]:
            c1 = preps.get(f"{gname}_K{K}_k1", {})
            c2 = preps.get(f"{gname}_K{K}_k2", {})
            c3 = preps.get(f"{gname}_K{K}_k3", {})
            if c1:
                lines.append(f"| {label} | {c1['F']*100:.2f}% ({c1['leak']*100:.2f}%) | {c2['F']*100:.2f}% ({c2['leak']*100:.2f}%) | {c3['F']*100:.2f}% ({c3['leak']*100:.2f}%) |")

    lines += [
        "",
        "## 3. Completed Circuit Runs (`n = 6` and `n = 8`, <= 5 Total Blocks Including State Prep)",
        "",
        "| Tag | `n` | Schedule | Gate | `N_train` | Relabel | Val Med / B90 | Test Med / B90 (`1/p`) | `NF_chk` Med / B90 | Mean Leak |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tag, J in sorted(runs.items(), key=lambda kv: (kv[1].get("args", {}).get("n", 0), kv[0])):
        lines.append(fmt_row(tag, J))

    open(OUT_MD, "w").write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD} with {len(runs)} runs.")

if __name__ == "__main__":
    main()

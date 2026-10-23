"""Generate /workspace/qumode_sep/oct_explore/results/OCT23_PART2_MULTICAV.md from finished Part 2 JSONs."""
import os, json, glob, time

RES_DIR = "/workspace/qumode_sep/oct_explore/results/oct23_exp_part2"
OUT_MD = "/workspace/qumode_sep/oct_explore/results/OCT23_PART2_MULTICAV.md"

def load_all():
    out = {}
    for p in sorted(glob.glob(os.path.join(RES_DIR, "*.json"))):
        name = os.path.basename(p)[:-5]
        try:
            out[name] = json.load(open(p))
        except Exception:
            pass
    return out

def fmt_cell(d):
    if not d:
        return "-"
    return f"**{d['median_inv_p']:.2f} / {d['B90_inv_p']:.2f}** (z:{d['zero_oracle_med']:.0f}, lk:{d['mean_leak']*100:.1f}%)"

def main():
    runs = load_all()
    lines = [
        "# Oct 23 Part 2: Multi-Cavity 4-SAT Scaling (`C = 2..5` Cavities, `n = 8, 12, 16, 20`, Physical SNAP+D + Global Cost + Pure GNN)",
        "",
        f"*Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')} (UTC+8). Every cavity holds `K = 4` bits in `D = 16` computational Fock levels (`0..15`, Gray code); all SNAP gates restricted to `0..15` (`0` phase on leaked levels `f >= 16`); displacements `D(alpha)` computed in `N_Fock = 40` and projected onto `0..15` at each layer (strictly no refocusing from leaked levels). Zero classical pre-solving (`relabel = none`, `fv[:,:,3:7] = 0`).*",
        "",
        "## 1. Classical WalkSATlm & Ideal-Mixer Reference Baselines on the 100 Unseen Test Formulas (`TEB = 52e9`)",
        "",
        "| Method | `n = 8` (`C=2`) Med / B90 | `n = 12` (`C=3`) Med / B90 | `n = 16` (`C=4`) Med / B90 | `n = 20` (`C=5`) Med / B90 | Growth `b^n` (Med / B90) |",
        "|---|---|---|---|---|---|",
        "| **WalkSATlm (lookups)** (`p=0.30`) | **14.0 / 29.4** | **32.5 / 76.1** | **71.0 / 147.0** | **110.0 / 290.0** | **1.191^n / 1.207^n** |",
        "| **WalkSATlm (tries)** (`p=0.30`) | **2.01 / 3.74** | **2.92 / 6.28** | **4.55 / 9.09** | **5.63 / 14.20** | **1.092^n / 1.116^n** |",
        "| *(ref)* Ideal Mixer QAOA L4 (`OCT23_DATA.md`) | 5.78 / 10.3 | 12.8 / 33.1 | 34.1 / 84.9 | 77.6 / 198.0 | 1.245^n / 1.278^n |",
        "| *(ref)* Ideal Mixer Global Cost L4 (`OCT23_DATA.md`) | 5.27 / 9.12 | 11.5 / 26.5 | 31.7 / 69.7 | 66.1 / 172.0 | 1.240^n / 1.277^n |",
        "",
        "## 2. Physical SNAP+D Multi-Cavity Scaling Runs (`n = 8, 12` Train -> `n = 8, 12, 16, 20` Unseen Test)",
        "",
        "| Tag | Mode (`Arch`) | `L` (`k_blk`) | Cost | Val12 Med / B90 | `n = 8` Med / B90 | `n = 12` Med / B90 | `n = 16` Med / B90 | `n = 20` Med / B90 | Growth `b^n` (Med / B90) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tag, J in sorted(runs.items(), key=lambda kv: (kv[1].get("args", {}).get("L", 0), kv[0])):
        a = J.get("args", {})
        te = J.get("test", {})
        v12 = J.get("val12", {})
        sc = J.get("scaling", {})
        mode_str = "Shared BFGS" if a.get("mode") == "shared" else f"Pure GNN (`{a.get('arch')}`)"
        cw_str = "clause_w" if a.get("clause_w") else "fixed_cost"
        lines.append(
            f"| `{tag}` | {mode_str} | `L={a.get('L')}` (`k={a.get('pairs')}`) | `{cw_str}` | {v12.get('median_inv_p',0):.2f} / {v12.get('B90_inv_p',0):.2f} | "
            f"{fmt_cell(te.get('8'))} | {fmt_cell(te.get('12'))} | {fmt_cell(te.get('16'))} | {fmt_cell(te.get('20'))} | "
            f"**{sc.get('b_median',0):.3f}^n / {sc.get('b_B90',0):.3f}^n** |"
        )

    open(OUT_MD, "w").write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD} with {len(runs)} runs.")

if __name__ == "__main__":
    main()

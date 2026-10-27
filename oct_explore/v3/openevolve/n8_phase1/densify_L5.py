#!/usr/bin/env python3
"""L=5-only densify around A_L5_n10000 (Med 38.3) — push toward WS lookups ~14.
Zach pivot: no L7/L9; focus ntrain~10k, H/rounds/steps/lr, leakpen/tailw, seeds.
"""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path

RUN = Path("/workspace/qumode_sep/oct_explore/v3/openevolve/runs/n8_phase1")
sys.path.insert(0, str(RUN))
spec = importlib.util.spec_from_file_location("rp", RUN / "run_phase1.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

def _add(q, seen, nt, phase, ov, extra):
    cid = rp.cfg_id("A", 5, nt, extra)
    if cid in seen or rp.is_done(cid):
        return
    seen.add(cid)
    q.append((cid, "A", 5, nt, phase, ov))

def l5_queue():
    q, seen = [], set()
    # --- Push harder toward 14: longer / hotter on winner recipe ---
    for steps, lr in (
        (2000, 1.1e-3), (2600, 1.1e-3), (2600, 8e-4), (3200, 8e-4),
        (3200, 1.1e-3), (4000, 6e-4), (4000, 1e-3), (4000, 1.1e-3),
    ):
        extra = f"L5_s{steps}_lr{str(lr).replace('-','m').replace('.','p')}"
        _add(q, seen, 10000, "L5_push", dict(steps=steps, lr=lr), extra)

    # --- ntrain near 10k ---
    for nt in (7000, 8000, 8500, 9000, 9500, 10000):
        if nt == 10000:
            continue  # exact winner already done as A_L5_n10000
        _add(q, seen, nt, "L5_ntrain", {}, f"L5_n{nt}")
    # ntrain + longer steps
    for nt in (8000, 9000):
        _add(q, seen, nt, "L5_ntrain", dict(steps=2600), f"L5_n{nt}_s2600")

    # --- capacity ---
    for H, r in ((64, 4), (96, 4), (96, 6), (128, 6), (128, 8), (160, 4), (160, 6), (192, 4)):
        _add(q, seen, 10000, "L5_cap", dict(H=H, rounds=r), f"L5_H{H}_r{r}")
    # capacity + longer steps
    for H, r, steps in ((128, 6, 2600), (160, 4, 2600), (160, 6, 2600), (192, 4, 2600), (96, 6, 2600)):
        _add(q, seen, 10000, "L5_cap", dict(H=H, rounds=r, steps=steps), f"L5_H{H}_r{r}_s{steps}")

    # --- lr at fixed steps=1400 and 2600 ---
    for lr in (5e-4, 6e-4, 8e-4, 1.0e-3, 1.2e-3, 1.4e-3, 1.6e-3):
        extra = f"L5_lr{str(lr).replace('-','m').replace('.','p')}"
        _add(q, seen, 10000, "L5_opt", dict(lr=lr), extra)
        _add(q, seen, 10000, "L5_opt", dict(lr=lr, steps=2600), extra + "_s2600")

    # --- regularize ---
    for leak in (0.06, 0.08, 0.10, 0.12, 0.16, 0.18, 0.22):
        _add(q, seen, 10000, "L5_reg", dict(leakpen=leak), f"L5_leak{leak}")
        _add(q, seen, 10000, "L5_reg", dict(leakpen=leak, steps=2600), f"L5_leak{leak}_s2600")
    for tw in (0.8, 1.0, 1.2, 1.4, 2.0, 2.5, 3.0):
        _add(q, seen, 10000, "L5_reg", dict(tailw=tw), f"L5_tail{tw}")
        _add(q, seen, 10000, "L5_reg", dict(tailw=tw, steps=2600), f"L5_tail{tw}_s2600")
    # combo reg
    for leak, tw in ((0.10, 1.2), (0.08, 1.0), (0.18, 2.0), (0.10, 2.0)):
        _add(q, seen, 10000, "L5_reg", dict(leakpen=leak, tailw=tw, steps=2600),
             f"L5_leak{leak}_tail{tw}_s2600")

    # --- seeds of winner (reproducibility + hope for better) ---
    for seed in (1, 2, 3, 4, 5):
        _add(q, seen, 10000, "L5_seed", dict(seed=seed), f"L5_seed{seed}")
        _add(q, seen, 10000, "L5_seed", dict(seed=seed, steps=2600), f"L5_seed{seed}_s2600")

    # --- best-guess combos toward 14 ---
    guesses = [
        (dict(H=128, rounds=6, steps=3200, lr=8e-4, leakpen=0.10, tailw=1.2), "guess_r6_long"),
        (dict(H=160, rounds=4, steps=3200, lr=1e-3, leakpen=0.12, tailw=1.6), "guess_H160_long"),
        (dict(H=128, rounds=4, steps=4000, lr=6e-4, leakpen=0.14, tailw=1.6), "guess_s4000_slow"),
        (dict(H=128, rounds=6, steps=4000, lr=8e-4, leakpen=0.10, tailw=1.0), "guess_deep"),
        (dict(H=160, rounds=6, steps=2600, lr=1.1e-3, leakpen=0.14, tailw=1.6), "guess_wide_deep"),
    ]
    for ov, extra in guesses:
        _add(q, seen, 10000, "L5_guess", ov, extra)

    return q

def main():
    rp.RUN.mkdir(parents=True, exist_ok=True)
    q = l5_queue()
    print(f"L5-ONLY DENSIFY queue={len(q)} (target WS Med~14; champ=38.3)", flush=True)
    (RUN / "DENSE_START").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f" L5_only n={len(q)}\n")
    (RUN / "PIVOT_L5").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + " skip L7/L9; densify L=5 only\n")
    rp.write_status()
    # prepend note
    st = (RUN / "STATUS.md").read_text()
    banner = (
        "\n## PIVOT (Zach)\n"
        "- **Stopped L=7/L=9** (A_L7_n10000 finished at Med 134≈floor before kill).\n"
        "- **L=5-only densify** around `A_L5_n10000` (38.3) → push toward WS ~14.\n"
        f"- Queue size at start: {len(q)}. Flag Med≲25 or new best≪38.\n"
    )
    if "PIVOT (Zach)" not in st:
        (RUN / "STATUS.md").write_text(st.rstrip() + "\n" + banner)
    for cid, tag, L, nt, phase, ov in q:
        row = rp.run_one(cid, tag, L, nt, phase=phase, overrides=ov)
        if row and row.get("status") == "done":
            med = row["test"]["median_inv_p"]
            if med <= 25 or med < 38.0:
                print(f"*** MILESTONE med={med:.2f} id={cid} ***", flush=True)
                with open(RUN / "MILESTONES.log", "a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} med={med:.3f} {cid}\n")
    rp.write_status()
    (RUN / "DENSE_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + " L5_only\n")
    print("L5 DENSIFY COMPLETE", flush=True)

if __name__ == "__main__":
    main()

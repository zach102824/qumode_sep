#!/usr/bin/env python3
"""Wide densify around winners A_L5_n10000 (38.3) and B_L3_n10000 (62.5).
Prioritized waves so high-signal lever tests run first.
"""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path

RUN = Path("/workspace/qumode_sep/oct_explore/v3/openevolve/runs/n8_phase1")
sys.path.insert(0, str(RUN))
spec = importlib.util.spec_from_file_location("rp", RUN / "run_phase1.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

def _add(q, seen, tag, L, nt, phase, ov, extra):
    cid = rp.cfg_id(tag, L, nt, extra)
    if cid in seen or rp.is_done(cid):
        return
    seen.add(cid)
    q.append((cid, tag, L, nt, phase, ov))

def wide_queue():
    """Ordered: wave1 high-signal, then wave2 fill."""
    q, seen = [], set()

    # ===== WAVE 0: push A_L5 toward WS~14 (highest priority) =====
    # Longer / hotter optimize on exact winner recipe
    for steps, lr in ((2000, 1.1e-3), (2600, 1.1e-3), (3200, 8e-4), (3200, 1.1e-3), (4000, 6e-4), (4000, 1e-3)):
        extra = f"push_steps{steps}_lr{str(lr).replace('-','m').replace('.','p')}"
        _add(q, seen, "A", 5, 10000, "dense_push", dict(steps=steps, lr=lr), extra)
    # Capacity bumps with enough steps
    for H, r, steps in ((128, 6, 2600), (160, 4, 2600), (160, 6, 2600), (192, 4, 2000), (96, 6, 2600)):
        _add(q, seen, "A", 5, 10000, "dense_push", dict(H=H, rounds=r, steps=steps), f"push_H{H}_r{r}_s{steps}")
    # L=4 and L=6 with A/10k longer train (fill depth gap)
    for L in (4, 6):
        _add(q, seen, "A", L, 10000, "dense_push", dict(steps=2600), f"push_L{L}_steps2600")
        _add(q, seen, "A", L, 10000, "dense_push", dict(steps=2000, rounds=6), f"push_L{L}_r6_s2000")
    # Mild reg tweaks on winner
    _add(q, seen, "A", 5, 10000, "dense_push", dict(steps=2600, leakpen=0.10, tailw=1.2), "push_reg_mild")
    _add(q, seen, "A", 5, 10000, "dense_push", dict(steps=2600, leakpen=0.18, tailw=2.0), "push_reg_mid")

    # ===== WAVE 1: what moves Med near A_L5_n10000 =====
    # L sweep with A/10k (core has 3,5,7,9 — fill 4,6 + confirm 5)
    for L in (4, 6):
        _add(q, seen, "A", L, 10000, "dense_L", {}, f"A_L{L}_n10000")
    # ntrain around 10k at L=5 A (core has 2k/5k/10k — try 7.5k, and 10k is done)
    _add(q, seen, "A", 5, 7500, "dense_ntrain", {}, "A_L5_n7500")
    _add(q, seen, "A", 5, 9000, "dense_ntrain", {}, "A_L5_n9000")
    # capacity at L=5 A 10k
    for H, r in ((96, 4), (128, 6), (160, 4), (128, 8), (64, 4)):
        _add(q, seen, "A", 5, 10000, "dense_cap", dict(H=H, rounds=r), f"A_H{H}_r{r}")
    # optimize
    for steps in (1000, 2000, 2600):
        _add(q, seen, "A", 5, 10000, "dense_opt", dict(steps=steps), f"A_steps{steps}")
    for lr in (6e-4, 8e-4, 1.0e-3, 1.4e-3):
        extra = f"A_lr{str(lr).replace('-','m').replace('.','p')}"
        _add(q, seen, "A", 5, 10000, "dense_opt", dict(lr=lr), extra)
    # regularize
    for leak in (0.08, 0.20):
        _add(q, seen, "A", 5, 10000, "dense_reg", dict(leakpen=leak), f"A_leak{leak}")
    for tw in (1.0, 2.5, 3.0):
        _add(q, seen, "A", 5, 10000, "dense_reg", dict(tailw=tw), f"A_tail{tw}")
    # seeds of winner
    for seed in (1, 2, 3):
        _add(q, seen, "A", 5, 10000, "dense_seed", dict(seed=seed), f"A_seed{seed}")

    # ===== WAVE 1b: neighborhood of B_L3_n10000 =====
    for L in (2, 4):
        _add(q, seen, "B", L, 10000, "dense_L", {}, f"B_L{L}_n10000")
    _add(q, seen, "B", 3, 7500, "dense_ntrain", {}, "B_L3_n7500")
    for H, r in ((64, 8), (80, 6), (96, 8), (80, 4), (128, 8)):
        _add(q, seen, "B", 3, 10000, "dense_cap", dict(H=H, rounds=r), f"B_H{H}_r{r}")
    for steps in (2000, 3200):
        _add(q, seen, "B", 3, 10000, "dense_opt", dict(steps=steps), f"B_steps{steps}")
    for lr in (4e-4, 1e-3):
        extra = f"B_lr{str(lr).replace('-','m').replace('.','p')}"
        _add(q, seen, "B", 3, 10000, "dense_opt", dict(lr=lr), extra)
    for leak in (0.08,):
        _add(q, seen, "B", 3, 10000, "dense_reg", dict(leakpen=leak), f"B_leak{leak}")
    for tw in (1.6, 4.0):
        _add(q, seen, "B", 3, 10000, "dense_reg", dict(tailw=tw), f"B_tail{tw}")
    for seed in (1, 2):
        _add(q, seen, "B", 3, 10000, "dense_seed", dict(seed=seed), f"B_seed{seed}")

    # ===== WAVE 2: cross / transfer =====
    # A recipe at L=3,4,6 with 10k; B recipe at L=4,5 already mostly in core
    _add(q, seen, "A", 4, 10000, "dense_cross", dict(steps=2000), "A_L4_steps2000")
    _add(q, seen, "A", 6, 10000, "dense_cross", dict(steps=2000, rounds=6), "A_L6_r6_steps2000")
    _add(q, seen, "A", 5, 10000, "dense_cross", dict(H=128, rounds=6, steps=2000, lr=1e-3), "A_bestguess")
    # Fix A_L5 data hunger: longer train at 5k? (maybe steps help) 
    _add(q, seen, "A", 5, 5000, "dense_data", dict(steps=2600), "A_L5_n5k_steps2600")
    _add(q, seen, "A", 5, 5000, "dense_data", dict(steps=2600, lr=6e-4), "A_L5_n5k_slow")

    return q

def main():
    rp.RUN.mkdir(parents=True, exist_ok=True)
    q = wide_queue()
    print(f"WIDE DENSIFY queue={len(q)}", flush=True)
    (RUN / "DENSE_START").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f" n={len(q)}\n")
    rp.write_status()
    for cid, tag, L, nt, phase, ov in q:
        rp.run_one(cid, tag, L, nt, phase=phase, overrides=ov)
    rp.write_status()
    (RUN / "DENSE_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    print("WIDE DENSIFY COMPLETE", flush=True)

if __name__ == "__main__":
    main()

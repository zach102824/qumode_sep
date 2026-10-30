# Multi-seed LBFGS polish — L32 projector champ

_Generated 2026-10-31 00:57:00 +0800 (Asia/Shanghai)_

Honesty: still `Uc=exp(-iγ Hc)`. No blind stencil.
Warm-start: tail **plain** `unsat_params.pt`. Soft pool `{9,12,15}` (n15=val).
8 seeds × 8 LBFGS rounds (noise 1e-3); top-3 by val S → scale1000.

Promote gate: n18 Med **or** B90 ≤ 98% of champ **and** n9/12/15 Med ≤ 105%.

## Champ baseline (pre-polish tail plain)

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.086 | 1.799 |
| 12 | 1.192 | 2.829 |
| 15 | 1.411 | 5.520 |
| 18 | 1.752 | 11.813 |

## Top-3 scale1000

| seed | val S | n9 | n12 | n15 | n18 | keep |
|------|-------|----|-----|-----|-----|------|
| **seed0** | 0.553 | **1.092/1.776** | **1.197/2.843** | **1.418/5.505** | **1.752/11.353** | **True** |
| seed4 | 0.551 | 1.090/1.783 | 1.196/2.863 | 1.408/5.581 | 1.744/11.446 | True |
| seed2 | 0.544 | 1.091/1.781 | 1.199/2.859 | 1.407/5.605 | 1.745/11.593 | False |

- **seed0**: keep — n18 B90 11.353 ≤ 11.577 (−3.9% vs 11.813); Med flat 1.752; n9/12/15 ok.
- **seed4**: keep — B90 11.446 ≤ 11.577; Med 1.744 (not ≤1.717).
- **seed2**: discard — B90 11.593 just shy of 11.577.

## Verdict

**PROMOTE** `seed0` → replace `unsat_params.pt` (backup: `unsat_params_tail_plain_backup.pt`).

| metric | old (tail plain) | **seed0** | Δ |
|--------|------------------|-----------|---|
| n18 Med | 1.752 | **1.752** | 0% |
| n18 B90 | 11.813 | **11.353** | −3.9% |
| b_Med | 1.055 | **1.054** | |
| b_B90 | 1.234 | **1.231** | |

Still renames \(H_c\). No blind-cost.

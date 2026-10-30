# Tail / CVaR retrain — soft-n15 L32 projector

_Generated 2026-10-31 00:30:58 +0800 (Asia/Shanghai)_

Honesty: still `Uc=exp(-iγ Hc)` (unsat projector). No blind stencil.
Warm-start: soft-n15 **t91215** `unsat_params.pt`. Train soft pool `{9,12,15}` (n15=val), L=32, Adam+LBFGS, 120 steps/arm.

Promote gate: n18 B90 ≤ 98% of champ **13.704** (=≤13.430) **and** n18 Med ≤ 103% of **1.884** **and** n9/12/15 Med ≤ 105% of champ.

## Champ baseline (pre-retrain)

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.100 | 1.759 |
| 12 | 1.231 | 2.919 |
| 15 | 1.483 | 5.847 |
| 18 | 1.884 | 13.704 |

## Arms (scale1000 holdout)

| arm | n9 | n12 | n15 | n18 | keep |
|-----|----|-----|-----|-----|------|
| **plain** | **1.086/1.799** | **1.192/2.829** | **1.411/5.520** | **1.752/11.813** | **True** |
| lme_tau1 | 1.103/1.760 | 1.235/2.913 | 1.489/5.771 | 1.884/13.589 | False |
| lme_tau2 | 1.101/1.764 | 1.229/2.896 | 1.476/5.861 | 1.873/13.656 | False |
| cvar_q80 | 1.102/1.773 | 1.236/2.902 | 1.485/5.785 | 1.884/13.745 | False |

- **plain**: keep — n18 B90 11.813 vs 13.704 (≤98%); Med 1.752 vs 1.884 (≤103%); n9/12/15 Med ok.
- **lme_tau1**: discard — B90 13.589 not ≤13.430 (≈−0.8% vs champ, shy of 2% gate).
- **lme_tau2**: discard — B90 13.656 not ≤13.430.
- **cvar_q80**: discard — B90 13.745 worse than champ.

## Verdict

**PROMOTE** `plain` → replace `unsat_params.pt` (backup: `unsat_params_softn15_t91215_backup.pt`).
Params also at `tail_retrain/promoted_params.pt` / `plain_params.pt`.

| metric | old champ | **plain** | Δ |
|--------|-----------|-----------|---|
| n18 Med | 1.884 | **1.752** | −7.0% |
| n18 B90 | 13.704 | **11.813** | −13.8% |
| b_Med (9→18) | ~1.062 | **1.055** | |
| b_B90 (9→18) | ~1.257 | **1.234** | |

Tail losses (logmeanexp / CVaR) did **not** beat mean-log warm retrain under matched budget; plain continued soft-n15 Adam and crushed the holdout tail.

## Note

Still renames \(H_c\). No blind-cost revival.

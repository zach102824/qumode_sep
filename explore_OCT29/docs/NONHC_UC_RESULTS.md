# Non-Hc Uc probes — verdict

_Generated 2026-10-31 10:17:50 (Asia/Shanghai)_

Ask (Zach): different Uc that does **not** secretly encode Hc.
Champ projector L44_cvar / Arm C densify = **renames_Hc**; cited only as baseline.

## Feature / algebra docs

### `surrogate_pair`

From clauses build signed pairwise J_ij += s_i*s_j for every pair in a clause. Diagonal cost E_surr[z] = sum_{i<j} J_ij * z_i * z_j with z_k=1-2x_k. This is a 2-body Ising surrogate of 4-SAT — NOT the unsat count Hc=sum_c C_c, and not a monotone rescaling of Hc (see honesty stats).

### `noncomm_xz`

Per cavity: U = exp(-iγ sum_j deg_j X_j) @ exp(-iγ sum_j bias_j Z_j), deg_j = clause incidence of bit j, bias_j = (#pos - #neg)/m. X and Z on the same bit do not commute → product ≠ any densified Z-phase of Hc. No clause projectors.

### `wrong_graph`

Fixed-per-n RNG perm π of variable labels; rewrite each clause v→π(v); cost = densified unsat of scrambled formula E_wrong; score with true E. Does not reconstruct true clauses unless shared SNAP+D magically inverts π (ablation vs trunc_Hc).

### Algebra check (`noncomm_xz`)

```json
{
  "gamma": 0.37,
  "frobenius_UxUz_minus_UzUx": 0.7397226319712411,
  "commute": false,
  "offdiag_frobenius_UxUz": 1.022802852519424,
  "is_diagonal": false,
  "equals_densified_Uc": false,
  "note": "UxUz non-diagonal & non-commuting \u2192 cannot equal exp(-i\u03b3 Hc) densified"
}
```

## Verdict table (L=16, train{9,12}, W=H^{⊗3} BinH)

| arm | =exp(Hc)? | n9 Med/B90 | n12 | n15 | vs ArmC n15 | vs trunc_Hc n15 | verdict |
|-----|-----------|------------|-----|-----|-------------|-----------------|---------|
| Arm C densify | YES | 1.441/3.314 | 1.991/6.252 | 2.921/10.959 | ×1.0000 | ×65.7912 | baseline (=Hc densify) |
| projector (=Hc) | YES | 1.406/3.288 | 1.864/6.106 | 2.754/10.244 | ×1.0606 | ×69.7807 | baseline (renames_Hc) |
| trunc_Hc (prior) | NO (partial) | 20.796/64.953 | 63.160/214.865 | 192.176/912.036 | ×0.0152 | ×1.0000 | baseline (prior FAIL vs ArmC) |
| surrogate_pair | NO | 209.948/1869.881 | 1656.957/11049.795 | 10576.639/68167.097 | ×0.0003 | ×0.0182 | KILL (Med=10576.6 ≱ trunc_Hc 192.2; ≫ ArmC 2.921) |
| noncomm_xz | NO | 646.940/11963.100 | 3754.804/91739.700 | 24479.091/857176.081 | ×0.0001 | ×0.0079 | KILL (Med=24479.1 ≱ trunc_Hc 192.2; ≫ ArmC 2.921) |
| wrong_graph | NO | 2444.996/32581.963 | 18519.756/305028.641 | 148343.797/1460528.625 | ×0.0000 | ×0.0013 | KILL (Med=148343.8 ≱ trunc_Hc 192.2; ≫ ArmC 2.921) |

## Interpretation

- **surrogate_pair**: KILL (Med=10576.6 ≱ trunc_Hc 192.2; ≫ ArmC 2.921) — =exp(Hc)? **NO**. From clauses build signed pairwise J_ij += s_i*s_j for every pair in a clause. Diagonal cost E_surr[z] = sum_{i<j} J_ij …
- **noncomm_xz**: KILL (Med=24479.1 ≱ trunc_Hc 192.2; ≫ ArmC 2.921) — =exp(Hc)? **NO**. Per cavity: U = exp(-iγ sum_j deg_j X_j) @ exp(-iγ sum_j bias_j Z_j), deg_j = clause incidence of bit j, bias_j = (#pos …
- **wrong_graph**: KILL (Med=148343.8 ≱ trunc_Hc 192.2; ≫ ArmC 2.921) — =exp(Hc)? **NO**. Fixed-per-n RNG perm π of variable labels; rewrite each clause v→π(v); cost = densified unsat of scrambled formula E_wro…

## Artifacts

- Out: `/workspace/qumode_sep_fresh/explore_OCT29/runs/nonhc_Uc`
- Script: `scripts/probe_nonhc_Uc.py`
- JSON: `runs/nonhc_Uc/results.json`

ArmC L44 fair-bar (PID pad) left untouched.

## One-line verdicts

1. **surrogate_pair** — KILL. Pairwise Ising surrogate (≠Hc, affine R²~0.33–0.43) n15 Med≈10577 ≫ trunc_Hc 192 / ArmC 2.92. B=200 screen; not promoted.
2. **noncomm_xz** — KILL. Non-commuting local XZ gadgets (Frobenius[UxUz−UzUx]≈0.74, off-diag≈1.02 ⇒ not densified Uc) n15 Med≈24479. B=200; not promoted.
3. **wrong_graph** — KILL. Scrambled-var clauses as cost (rank corr vs true Hc ≈0) n15 Med≈1.5e5; SNAP+D does not invert π. Worse than trunc_Hc.

**Bottom line:** still no competitive non-Hc Uc. Hardware story remains frozen clause-projector library (=Uc=exp(−iγ Hc)).

Screen B=200 (promote threshold: beat trunc_Hc n15). Honesty samples in `results.json`.

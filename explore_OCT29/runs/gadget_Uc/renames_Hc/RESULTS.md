# renames_Hc — CRITICAL FIX soft-n18 from locked champ

_Generated 2026-10-31 02:33:30 +0800 (Asia/Shanghai)_

Honesty: still `Uc=exp(-iγ Hc)`. No blind-cost.

## CRITICAL — prior soft-n18 table INVALID

The earlier soft-n18 numbers (n9–18 ≈ **2.241/5.200, 3.555/10.412, 5.760/18.361, 9.802/38.644**)
were trained from a **bad short retrain seed**, **not** locked LBFGS seed0.
**Do not cite for promote / paper.** Artifacts kept under `INVALID_*` / old RESULTS sections only as cautionary.

## Locked champ restored + verified

Source: `lbfgs_polish/seed0_params.pt` (= `promoted_params.pt`) → `unsat_params.pt`.

| n | sealed | measured | ΔMed | ΔB90 |
|---|--------|----------|------|------|
| 9 | 1.092/1.776 | 1.092/1.776 | +0.0004 | +0.0003 |
| 12 | 1.197/2.843 | 1.197/2.843 | -0.0002 | +0.0005 |
| 15 | 1.418/5.505 | 1.418/5.505 | +0.0004 | -0.0003 |
| 18 | 1.752/11.353 | 1.752/11.353 | +0.0002 | -0.0003 |

Match within rounding (max |ΔMed|≈0.0004). b_Med≈**1.0543**.

## Soft-n18 FROM locked champ (this fix)

Train soft-pool `{9,12,15,18}` (n15/n18 = val; scale1000 = holdout).
Gate vs **locked sealed**: n18 Med≤98% **or** B90≤98%; AND n9/12/15 Med≤105%.

| arm | n9 | n12 | n15 | n18 | keep |
|-----|----|-----|-----|-----|------|
| t9121518 | 1.107/1.751 | 1.220/2.753 | 1.440/5.483 | 1.772/11.249 | 0 |
| t9121518_w1 | 1.098/1.719 | 1.197/2.779 | 1.395/5.460 | 1.709/11.107 | 1 |

**Winner / PROMOTED:** `t9121518_w1`

| n | locked | soft-n18 promoted | Δ |
|---|--------|-------------------|---|
| 9 | 1.092/1.776 | **1.098/1.719** | +0.006/-0.057 |
| 12 | 1.197/2.843 | **1.197/2.779** | +0.000/-0.064 |
| 15 | 1.418/5.505 | **1.395/5.460** | -0.023/-0.045 |
| 18 | 1.752/11.353 | **1.709/11.107** | -0.043/-0.246 |

b_Med≈**1.0506**, b_B90≈**1.2325** (was locked 1.054/1.231).

Zero-shot honesty note: locked LBFGS never saw n18 train; soft-n18 uses n18 **val** (B=16)
in the soft pool. scale1000 (B=1000, different seeds) remains the sealed holdout.
Promote is real on holdout but is **not** pure zero-shot n18 anymore.

Champ path: `unsat_params.pt` ← soft-n18; backup `unsat_params_pre_soft18_backup.pt` (= locked seed0).

## SNAP φ noise (prior wave)

Prior SNAP φ table used the **invalid retrain seed** — qualitative only (σ≤2% mild).
Re-run on promoted champ deferred to next cheap probe.

## WalkSAT (still valid; unrelated to soft-n18 bug)

| n | flips Med/B90 | evals Med/B90 |
|---|---------------|---------------|
| 9 | 12.5/40.1 | 51.0/161.4 |
| 12 | 26.5/65.9 | 107.0/264.6 |
| 15 | 43.8/128.1 | 176.0/513.6 |
| 18 | 71.5/203.6 | 287.0/815.4 |

# SNAP φ noise on soft-n18 promoted champ

_Generated 2026-10-31 02:34:37 +0800 (Asia/Shanghai)_

Honesty: still `Uc=exp(-iγ Hc)`. Replaces prior SNAP table that used INVALID retrain seed.

| σ | n9 Med/B90 | n12 Med/B90 | vs σ=0 n9 |
|---|------------|-------------|---------|
| 0.00 | 1.098/1.719 | 1.197/2.779 | ×1.000 |
| 0.01 | 1.101/1.726 | 1.201/2.797 | ×1.002 |
| 0.02 | 1.131/1.763 | 1.242/2.855 | ×1.030 |
| 0.05 | 1.415/2.071 | 1.701/3.399 | ×1.288 |
| 0.10 | 1.756/3.074 | 2.261/6.166 | ×1.599 |

Verdict: compare to γ-jitter; σ≤2% typically mild.


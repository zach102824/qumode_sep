# Fixed γ (one per instance / shared) + deeper projector

_Generated 2026-10-29 18:23:51 (Asia/Shanghai)_

Circuit: `|0⟩→W→[U_Φ(γ) → W·SNAP₀(β_ℓ)·W†]^L` with **same γ every layer**; β from Fourier modes.

## Clarification

- **shared_gamma**: one γ for all instances (fixed-angle).
- **per_instance_gamma**: β frozen; γ L-BFGS per val instance (classical outer; not transferable alone).

## Refs

- Champ fourier_proj_L16: n15≈2.51, b≈1.08
- Fourier q3 @L20: n15≈5.99, b≈1.24

## Shared γ (fixed-angle)

Trained γ=4.103; L_train=6

| n \ L | 4 | 8 | 12 | 16 | 20 | 24 |
|-------|------|------|------|------|------|------|
| 9 | 7.51/27.5 | 7.47/18.1 | 10.5/21.5 | 14/20.4 | 28.3/39 | 29.4/103 |
| 12 | 21.5/37.2 | 17.6/43.3 | 22/38.9 | 49.3/124 | 80.2/544 | 121/244 |
| 15 | 54.4/104 | 40.3/73.5 | 64.5/114 | 148/267 | 438/825 | 686/2.98e+03 |
| 18 | 216/725 | 246/1.46e+03 | 1.46e+03/1.36e+04 | 1.01e+03/2.14e+03 | 2.39e+03/1.72e+04 | 4.32e+03/8.38e+03 |

| L | b_Med | b_B90 |
|---|-------|-------|
| 4 | 1.443 | 1.435 |
| 8 | 1.458 | 1.579 |
| 12 | 1.697 | 1.976 |
| 16 | 1.592 | 1.633 |
| 20 | 1.649 | 1.865 |
| 24 | 1.745 | 1.688 |

## Per-instance γ (opt at eval)

_n18 skipped (CPU); through n15 only_

| n \ L | 4 | 8 | 12 | 16 | 20 |
|-------|------|------|------|------|------|
| 9 | 7.47/27.5 | 7.31/17.9 | 8.73/19.7 | 13/20.2 | 19.9/30.8 |
| 12 | 21.4/36.7 | 16.2/43.2 | 19/38 | 35/97.1 | 43.3/117 |
| 15 | 54.3/102 | 39.3/72.8 | 62/108 | 122/266 | 128/309 |

| L | b_Med | b_B90 |
|---|-------|-------|
| 4 | 1.392 | 1.245 |
| 8 | 1.323 | 1.264 |
| 12 | 1.386 | 1.328 |
| 16 | 1.453 | 1.537 |
| 20 | 1.364 | 1.469 |

## Verdict

Shared one-γ: n15 L8=40.3, L12=64.5, L24=686 (deeper than ~L8 makes Med worse). Per-instance γ: n15 L8=39.3, L20=128 — only marginally better than shared, still ≫ champ n15=2.51/b=1.08. **Need γ_ℓ (or Fourier) schedule** — one fixed γ per instance does not unlock good deep scaling.

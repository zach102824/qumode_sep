# Finite-shot library schedule — LBFGS seed0 champ

_Generated 2026-10-31 01:04:55 +0800 (Asia/Shanghai)_

Honesty: still `Uc=exp(-iγ Hc)` via frozen SNAP-mask library schedule. No blind stencil.

Analytic amp: `inv_p = 1/p_succ` (computational success mass).
Finite shots: `k ~ Binomial(S, p_succ)`, `p̂=k/S`, `inv̂=1/max(p̂, 1/(S+1))`.

## Champ analytic baseline (sealed)

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.092 | 1.776 |
| 12 | 1.197 | 2.843 |
| 15 | 1.418 | 5.505 |
| 18 | 1.752 | 11.353 |

## Finite-shot Med/B90 vs analytic

| n | shots | Med | B90 | ΔMed% | ΔB90% | zero_frac | B |
|---|-------|-----|-----|-------|-------|-----------|---|
| 9 | 10 | 1.111 | 2.000 | +1.7% | +12.6% | 0.008 | 1000 |
| 9 | 50 | 1.087 | 1.724 | -0.5% | -2.9% | 0.003 | 1000 |
| 9 | 100 | 1.099 | 1.786 | +0.6% | +0.5% | 0.000 | 1000 |
| 9 | 500 | 1.091 | 1.751 | -0.2% | -1.4% | 0.000 | 1000 |
| 9 | ∞ analytic | 1.092 | 1.776 | 0.0% | 0.0% | 0.000 | 1000 |
| 12 | 10 | 1.250 | 3.333 | +4.4% | +17.2% | 0.031 | 1000 |
| 12 | 50 | 1.220 | 2.941 | +1.9% | +3.4% | 0.008 | 1000 |
| 12 | 100 | 1.205 | 3.030 | +0.7% | +6.6% | 0.007 | 1000 |
| 12 | 500 | 1.196 | 2.943 | -0.1% | +3.5% | 0.003 | 1000 |
| 12 | ∞ analytic | 1.197 | 2.843 | 0.0% | 0.0% | 0.000 | 1000 |
| 15 | 10 | 1.429 | 5.000 | +0.7% | -9.2% | 0.046 | 1000 |
| 15 | 50 | 1.429 | 5.556 | +0.7% | +0.9% | 0.013 | 1000 |
| 15 | 100 | 1.408 | 5.263 | -0.7% | -4.4% | 0.008 | 1000 |
| 15 | 500 | 1.412 | 5.325 | -0.4% | -3.3% | 0.002 | 1000 |
| 15 | ∞ analytic | 1.418 | 5.505 | 0.0% | 0.0% | 0.000 | 1000 |
| 18 | 10 | 1.667 | 10.000 | -4.9% | -11.9% | 0.098 | 1000 |
| 18 | 50 | 1.724 | 10.000 | -1.6% | -11.9% | 0.024 | 1000 |
| 18 | 100 | 1.724 | 11.250 | -1.6% | -0.9% | 0.017 | 1000 |
| 18 | 500 | 1.745 | 11.905 | -0.4% | +4.9% | 0.005 | 1000 |
| 18 | ∞ analytic | 1.752 | 11.353 | 0.0% | 0.0% | 0.000 | 1000 |

## Sim → exp gap on frozen SNAP masks

Library = 26 local projectors as SNAP masks (θ_f = γ·mask). Single-cavity compile:

- Fid SNAP↔U_Z: min=1.000000 (n=24) — **exact** within numerics.
- Fid W†SNAP W↔U_X: min=1.000000000000 — **exact** BinH sandwich.
- Fallback SNAP·D fit to U_X (no native W): min=0.6626, mean=0.9022 — **sim≠exp** if W unavailable.

| γ | SNAP·D Fid min | mean |
|---|----------------|------|
| 0.2 | 0.9928 | 0.9950 |
| 0.5 | 0.9560 | 0.9694 |
| 1.0 | 0.8341 | 0.8838 |
| 1.5 | 0.6626 | 0.7608 |

Two-cavity AND-phase: local SNAP⊗SNAP Fid≪1; **joint SNAP** Fid=1 (see `grape_library/COMPILE_TWOCAV.md`, `pulse_budget/`). Cross-Kerr T already tabulated there — not re-run.


## Estimator caveats

- Floor `1/(S+1)` caps zero-count inv at S+1; at small S this **quantizes** Med/B90 onto `{S/k}` grid (e.g. B90=2.000, 3.333, 10.000).
- ΔMed% can be **negative** at small S (optimistic) when the discrete mode sits above true p_succ — not a free lunch; zero_frac and B90 tell the reliability story.
- Prefer S≥100 (n≤15) / S≥500 (n18) for amp-faithful Med; B90 needs more shots than Med.

### Takeaway

- With native W + joint SNAP: **unitary sim→exp gap ≈ 0** on frozen masks; remaining gap is **readout shot noise** (table above).
- Without W / without joint SNAP: pulse-compile fidelity gap dominates (γ-dependent).
- Finite S **distorts** Med/B90 via quantization + zero-counts; S≳500 recovers analytic Med within ~1% (n≤18); B90 still ±few %.


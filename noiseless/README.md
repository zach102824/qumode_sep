# noiseless/

Noiseless SPSA campaign: local ECD on diagonal pairs `(d–A)` / `(e–B)` plus a
**frozen** bus unitary `U` on `A⊗B`. Prep is not trained; `U` is never trained;
ECD parameters are randomly initialized each trial.

## Encoding

`|d⟩ ⊗ |e⟩ ⊗ |A⟩ ⊗ |B⟩`, Fock cutoff 8 → dim `2×2×8×8 = 256` (exact 8 bits).

Bit map (MSB-first): `(q_d, q_e | n_A[2:0] | n_B[2:0])`.

## Ansatz

One layer: `(ECD on A–d ‖ ECD on B–e) → U_fixed`, repeated `L*` times with the
same frozen `U` and fresh ECD params each layer (`8 L*` Cartesian parameters).

## Fixed-U library

`identity`, `bs_pi4`, `bs_pi2`, `cz_nm`, `snap_a_pi`, `snap_b_pi`, `ck_pi2`, `ck_pi4`, `jp`, … (see `unitaries.U_NAMES`)

## Cost / optimizer

Gibbs with `sampled_tail` η (no known `E_min` during opt). SPSA on ECD params
only. Default 200 steps; `a ∝ 1/√n_params` (baseline `a≈0.2` at `n_params=37`).

### Optional β-aware loss (opt-in)

Default is **Gibbs-only** (`--lambda1 0 --lambda 0`, no `--beta-max`): identical
to the pre-β-aware cost path.

When enabled:

```
L(x) = Gibbs(x; η) + λ1 Σ_i |β_i| + λ Σ_i max(|β_i| - β_max, 0)^2
```

β_i are the complex ECD displacements unpacked per layer (`β_d`, `β_e`).
Soft-cap weight is **λ** (`--lambda`; Python field `lam`). Legacy `--lambda3` is an alias.

```bash
# Gibbs-only (default)
python -m noiseless.run_u_sweep --u-names ck_pi4 --layers 4 --tag fleet_beta_aware_A0_ckpi4

# L1 on |β|
python -m noiseless.run_u_sweep --u-names ck_pi4 --layers 4 \
  --lambda1 0.05 --lambda 0 --tag fleet_beta_aware_A1_l1_0p05

# Fixed soft |β| cap
python -m noiseless.run_u_sweep --u-names ck_pi4 --layers 4 \
  --lambda 3 --beta-max 2.1 --tag fleet_adapt_B1_fixed_l3

# Adaptive soft-cap λ (opt-in; default OFF)
# Warm-up 25% of steps with λ=0, then every 25 steps raise/lower λ from
# frac(|β_i|>β_max) vs thresholds f_hi=0.20 / f_lo=0.05 (λ∈[0, lam_max]).
python -m noiseless.run_u_sweep --u-names ck_pi4 --layers 4 \
  --adapt-lambda --beta-max 2.1 --tag fleet_adapt_B2_adaptive
```

### Optimizer choice (`--optimizer`)

`--optimizer spsa` (default) is the SPSA loop above; old runs reproduce exactly.
`--optimizer bfgs` runs `scipy.optimize.minimize(method="BFGS")` with default
settings (finite-difference gradient, `maxiter=--steps`) from the same random x0
(drawn from the trial seed exactly as for SPSA) on the same cost (Gibbs + optional
λ terms). η is refreshed at the start and then after every 5th BFGS iteration
(same cadence as SPSA steps 1, 6, 11, ...). Records carry `optimizer`, true cost
evaluation count `nfev`, BFGS `nit`, `opt_status` and `opt_message`.
`--adapt-lambda` is not supported with BFGS.
`--optimizer spsa_adam` keeps the SPSA step (same η refresh, c_k, Rademacher rng draws,
2 evals/step + 1 final, so identical `nfev`) but replaces x -= a_k ĝ with a bias-corrected
Adam update (`--adam-lr`, default 0.05; β1 0.9, β2 0.999, eps 1e-8). Works with `--grow`
(fresh Adam state per stage). See `results/ADAM_SUMMARY.md`.

```bash
python -m noiseless.run_u_sweep --u-names jp --layers 4 --trials 25 --steps 200 \
  --workers 1 --lambda1 0 --optimizer bfgs --tag jp_bfgs_B0
```

### Cavity encoding (`--encoding`) and layer growth (`--grow`)

`--encoding binary` (default; old runs reproduce bit-for-bit) maps Fock n → the 3 cavity
bits as plain binary. `--encoding gray` uses the Gray code g = n ^ (n>>1) (transmon bits
unchanged); the logical Hamiltonian is unchanged, so the physical energy tensor is a
permutation of the binary one and the logical GS bitstring is identical. Records and
summaries carry `encoding`.

`--grow` trains L=`--grow-start` (default 1) from the usual random init, then appends a
probability-transparent LAST layer (β=0, θ=π, φ=0 for d and e: ECD(0)·R(π,0) = −i·I per
transmon, and a diagonal U such as `jp` preserves probabilities) plus a Gaussian kick
σ=`--grow-kick-sigma` (0.05) on the new layer's 8 params, retrains, and repeats up to each
`--layers` value. Each stage is a fresh SPSA run (gain `a` rescaled to the stage's
n_params, η controller restarted). Budget: `--steps` is the TOTAL split evenly over stages
(200 → 100/100, 67/67/66, 50×4) unless `--grow-steps-per-stage S` is given (S per stage,
total S×n_stages). Records carry per-stage metrics in `stages`.

```bash
python -m noiseless.run_u_sweep --u-names jp --layers 2,3,4 --trials 25 --steps 200 \
  --workers 8 --seed 20260917 --lambda1 0 --encoding gray --grow --grow-steps-per-stage 200 \
  --tag jp_enc_D_gray_grow_ps200
```

#### Growth + BFGS (`--optimizer bfgs --grow`, `--bfgs-eta-mode`)

With `--optimizer bfgs --grow` each stage is a BFGS run (finite-difference gradient,
gtol 1e-5) warm-started from the previous stage's x plus the transparent + kicked new layer;
the stage step count is the BFGS `maxiter` (use `--grow-steps-per-stage 500`). The η
controller restarts every stage, so η is always re-derived between stages.
`--bfgs-eta-mode` (BFGS only) picks how η is treated *within* a run:

- `callback` (default, legacy numerics): refresh every 5 BFGS iterations inside the run.
  BFGS's stored f/gradient then refer to a stale objective → the multiple-of-5
  "precision loss" stops seen in `results/BFGS_VS_SPSA_JP_SUMMARY.md`.
- `fixed`: refresh once from the stage's x0, then hold η (stationary objective).
- `restart`: BFGS to convergence at fixed η → refresh η (same sampled-tail controller /
  EMA) → if η moved > 1 % (rel.) restart BFGS from the result with the inverse Hessian
  carried over (scaled by η_old/η_new; identity if not positive definite). Stops on
  `eta_converged`, `x_stationary` (a restart made no iteration), `maxiter` (total BFGS
  iterations over all restarts), or 50 restarts. This is the recommended mode: it follows
  the η schedule without breaking BFGS's fixed-objective assumption.

BFGS stage records add `nit`, `opt_status`, `opt_message`, `wall_s` and (non-legacy modes)
`termination`, `n_restarts`, `etas`; records add `bfgs_eta_mode` and `opt_info`.

```bash
python -m noiseless.run_u_sweep --u-names jp --layers 4 --trials 25 --workers 8 \
  --seed 20260917 --lambda1 0 --encoding binary --optimizer bfgs --grow \
  --grow-steps-per-stage 500 --bfgs-eta-mode restart --tag jp_grow_bfgs_restart
```

## CLI

```bash
source /workspace/qumode/.venv/bin/activate
export PYTHONPATH=.

python -m pytest noiseless/tests tests/test_four_sat.py -q
python -m noiseless.run_u_sweep --smoke
python -m noiseless.run_u_sweep --ham-dir Hamiltonians/four_sat \
  --u-names all --layers 2,3,4 --trials 5 --steps 200 --workers 4 --tag fleet1
```

Results: `noiseless/results/`. Hamiltonians: `Hamiltonians/four_sat/` (8-qubit).

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

`identity`, `bs_pi4`, `bs_pi2`, `cz_nm`, `snap_a_pi`, `snap_b_pi`, `ck_pi2`, `ck_pi4`, … (see `unitaries.U_NAMES`)

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

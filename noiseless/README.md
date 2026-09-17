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

`identity`, `bs_pi4`, `bs_pi2`, `cz_nm`, `snap_a_pi`, `snap_b_pi`

## Cost / optimizer

Gibbs with `sampled_tail` η (no known `E_min` during opt). SPSA on ECD params
only. Default 200 steps; `a ∝ 1/√n_params` (baseline `a≈0.2` at `n_params=37`).

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

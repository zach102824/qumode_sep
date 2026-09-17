# noiseless/

Noiseless SPSA campaign: local ECD on diagonal pairs `(d–A)` and `(e–B)` plus a
**frozen** bus unitary `U` on `A⊗B`. No jointly trained prep; fixed `U` is never
trained; ECD parameters are randomly initialized each trial.

## Encoding

State: `|d⟩ ⊗ |e⟩ ⊗ |A⟩ ⊗ |B⟩` with Fock cutoff 8 → dim `2×2×8×8 = 256` (exact 8 bits).

Bit map (MSB-first): `(q_d, q_e | n_A bits[2:0] | n_B bits[2:0])`.

The physical middle bus qubit is omitted; the bus is an ideal unitary on `A⊗B` only.

## Ansatz

One layer: `(ECD on A–d ‖ ECD on B–e) → U_fixed`, repeated `L*` times with the
**same** frozen `U` and fresh ECD params each layer.

ECD params: 4 reals × 2 pairs × `L*` = `8 L*` (Cartesian: Reβ, Imβ, θ, φ).

## Fixed-U library

| name | action on `A⊗B` |
|------|-----------------|
| `identity` | `I` |
| `bs_pi4` | beamsplitter `θ=π/4` |
| `bs_pi2` | beamsplitter `θ=π/2` |
| `cz_nm` | `\|n,m⟩ ↦ (-1)^{nm} \|n,m⟩` |
| `snap_a_pi` | `\|n,m⟩ ↦ e^{-iπ n} \|n,m⟩` |
| `snap_b_pi` | `\|n,m⟩ ↦ e^{-iπ m} \|n,m⟩` |

## Cost / optimizer

Gibbs objective with `sampled_tail` η (histogram quantiles; no known `E_min`
during opt). SPSA on ECD params only. Default 200 steps; `a ∝ 1/√n_params`
(baseline `a≈0.2` at `n_params=37`).

Metric: ground-bitstring success (decoded 8-bit string == GS) and mean `p(GS)`.

## CLI

From repo root (venv with qutip/numpy/scipy):

```bash
source /workspace/qumode/.venv/bin/activate   # or equivalent
export PYTHONPATH=.

# unit tests
python -m pytest noiseless/tests -q

# smoke (synthetic H if NPZs not ready)
python -m noiseless.run_u_sweep --smoke --synthetic

# fleet
python -m noiseless.run_u_sweep \
  --ham-dir Hamiltonians/four_sat \
  --u-names all --layers 2,3,4 \
  --trials 5 --steps 200 --workers 2 --tag fleet1
```

Hamiltonians come from `Hamiltonians/four_sat/` (8-qubit NPZs).

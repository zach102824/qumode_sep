# Round 1 direction D

Shared-gamma native Nov_3rd shell (`G = Hc`) against independent-angle QAOA on the open-chain Pagano-like AF TFIM.

```text
H = sum_{i<j} |i-j|^{-1.1} Z_i Z_j + B sum_i X_i
```

`B < 0` is chosen separately at N=6 and N=12 by maximizing the half-chain entropy of the full-space ground state. That field is this run's numerical realization, not a published Pagano value.

## Ansatz

Logical frame, rightmost factor acts first:

```text
|+> -> [exp(-i gamma Hc) -> M_l ]^L
M_l = exp(-i sum_cavities [c1 sum X + c2 sum XX + c3 XXX])
```

Shell phases follow the contract,

```text
phi(k) = c1(3-2k) + c2[((3-2k)^2-3)/2] + c3(-1)^k
```

with bit 1 corresponding to `Z = -1`. One gamma is shared by every cost block. Shell coefficients may change with the layer. There is no split gamma and no per-layer theta.

QAOA uses an independent `(gamma_l, beta_l)` and the mixer `exp(-i beta_l sum X)`. `B` is in the energy, not in the mixer. At L=1 that QAOA ansatz is exactly the shared-beta ablation.

## Run

From this directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 run_round1.py --budget-min 80
```

Algebra checks run first and stop the job if the physical and logical circuits disagree. Results land in `results/`: targets and hashes, angles, state vectors, per-start energies, convergence traces, figures, `summary.json`, and `RESULTS.md`.

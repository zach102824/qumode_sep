# Family C — pure distinct-index 4-body ferro / antiferro

This folder is **not** a random p-spin glass. Every quadruple has the same coupling.

```
J = 1
a_N = N / C(N, 4) = 24 / ((N-1)(N-2)(N-3))
H_C = s * J * a_N * sum_{i<j<k<l} Z_i Z_j Z_k Z_l
H   = H_C - Gamma * sum_i X_i
s   = -1 ferro,  +1 antiferro
```

`a_N` is extensive for the fully polarized configurations: their cost is exactly `s*J*N`.
The same polarized cost is shared by the collective control

```
H_C^ctrl = s * J * N * m_z^4 = s * J * M^4 / N^3
```

which is **only** a control. Expanding `M^4` with `Z_i^2 = I` produces repeated-index
contractions, so `N m_z^4` is not the distinct four-body sum. See `model.py` for the
operator identity and `validate.py` for the enumeration check.

## Reproduce

```
python3 validate.py
python3 run_round1.py
python3 plot_round1.py
```

Primary circuit: independent `(gamma_l, beta_l)`, cost unitary then `sum X` mixer,
initial `| + >`, **energy** objective. Shared-gamma, an L=1 fidelity maximization,
and a digitized linear anneal are stored under separate `method` values in
`results/round1/qaoa.csv`.

Angle periods are derived from the normalized spectrum (`T_gamma` in the outputs).
The projective period of each `beta` is `pi`. Raw optimizer angles are kept;
canonical folding is reporting-only and is not fed back into the bootstrap.

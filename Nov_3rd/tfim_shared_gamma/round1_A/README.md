# Round 1 direction A

Shared-gamma Nov_3rd ansatz on the open-chain power-law TFIM, with one
coupling-informed two-body feature.

```text
H = Hc + B sum_i X_i
Hc = sum_{i<j} J_ij Z_i Z_j
J_ij = |i-j|^(-1.1)
G = Hc + thetaK * sK * F_K
F_K = sum_{j<k} K_jk Z_j Z_k
K_jk = sum_{i != j,k} J_ij J_ik
sK = rms(J) / rms(K)
```

`gamma` and `thetaK` are shared by every layer. Each layer has three shell
controls. The coefficient of `Hc` is fixed at 1.

From the repository root:

```bash
python3 -m Nov_3rd.tfim_shared_gamma.round1_A.run_round1 --budget-min 80
```

Validation alone:

```bash
python3 -m Nov_3rd.tfim_shared_gamma.round1_A.validate
```

Outputs land in `results/`. `summary.json` is the machine-readable record.
`target_n{6,12}.json` holds B, the coupling table, the spectrum, the entropy
curve, and the Hamiltonian hash. `target_n{6,12}.npz` holds the ground state
and the normalized K matrices.

The Hamiltonian hash is SHA-256 of canonical JSON with `B` to 10 decimals and
the upper triangle of `J` to 12-digit scientific notation. State hashes are
SHA-256 of the canonical-phase float64 vector `[real; imag]`, plus a
decimals=10 variant for cross-agent comparison.

Training minimizes energy only. Fidelity is computed after the fact and is
never used to pick angles or features. N=6 to N=12 transfer freezes angles.
The primary gamma rescaling is `kappa_6 / kappa_12`. That transfer is not the
retrained N=12 run.

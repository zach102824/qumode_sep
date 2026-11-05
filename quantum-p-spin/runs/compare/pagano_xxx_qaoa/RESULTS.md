# Pagano-like AF XXX QAOA (XX→XXX)

Paper: [arXiv:1906.02700](https://arxiv.org/abs/1906.02700)

## Hamiltonian

```
H = Σ_{i<j<k} J_ijk X_i X_j X_k + B Σ_i Y_i
J_ijk = c / r_ijk^α, r=max(|i-j|,|i-k|,|j-k|) open 1D, c so ⟨J⟩_{r=2}=J0=1
|ψ0⟩ = |+⟩_y^{⊗N}
QAOA: ∏ exp(-i β_k H_B/J0) exp(-i γ_k H_A/J0) |ψ0⟩
η = (E-Emax)/(Egs-Emax)
```

## p=1 η* (SV)

| n \\ B/J0 | -0.3 |
|---|---|
| **6** | 0.6305 |

Paper target near |B/J0|≈0.3: p=1 η* ≳ 0.94.

## Detail

| n | B/J0 | Egs | E_QAOA | E_init | S_half | η* | η_init | β* | γ* | F_gs |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | -0.3 | -12.64342 | -3.30053 | -1.80000 | 0.0000 | 0.6305 | 0.5712 | 0.8674 | 0.1467 | 0.0892 |

## Half-cut S (GS)

| n | S@-0.3 |
|---|---|
| 6 | 0.0000 |

- |⟨ψ0|HB_gs⟩|² min = 1.000000
- avg_NN = 1.000000

Generated 2026-11-05 23:38:36 CST

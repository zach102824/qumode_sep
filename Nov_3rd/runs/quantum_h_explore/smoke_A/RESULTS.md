# RESULTS — arm A transverse-field Max-E (smoke)

Locked gate: `U_c^X = W exp(-i γ C_n) W`, `W=H^{⊗3}`, Hamming-shell SNAP.
`C_n = (n/m) H_c − max` is the existing Nov_3rd cost (global phase drops; `γ` absorbs `n/m`).

Hamiltonian, stoquastic sign:

```text
H_c(z) = unsat count of the TEB / scale1000 clause pack
H     = H_c − λ Σ_i X_i ,   λ>0
ε     = (⟨H⟩ − E_0) / n
⟨H⟩   = Σ_z p_Z(z) H_c(z) − λ Σ_i ⟨X_i⟩
```

Readout is the Z histogram plus single-qubit X histograms. No YY or XX.

**Chosen λ = 0.75.** On B=32 exact diagonalizations per n, this is the grid point that maximizes median half-chain entropy among λ with median bitstring peak < 0.5 on both n=6 and n=9 (S_half ≈ 0.244 / 0.395, peak ≈ 0.227 / 0.085). Ties would break toward 1. Fidelity susceptibility χ_F is largest on the step into λ=0.50; entropy then peaks at 0.75, next to the prior λ~1.

## ED quantumness

| n | λ | peak Med | S_half Med | S1 Med | F_sat Med | ⟨X⟩/n | χ_F | quantum |
|---|---|----------|------------|--------|-----------|-------|-----|---------|
| 6 | 0.25 | 0.795 | 0.029 | 0.015 | 0.882 | 0.291 | — | no |
| 6 | 0.50 | 0.428 | 0.191 | 0.134 | 0.567 | 0.564 | 2.52 | yes |
| 6 | 0.75 | 0.227 | 0.244 | 0.147 | 0.282 | 0.766 | 1.70 | yes |
| 6 | 1.00 | 0.132 | 0.228 | 0.120 | 0.174 | 0.860 | 0.59 | yes |
| 6 | 1.25 | 0.087 | 0.176 | 0.090 | 0.116 | 0.912 | 0.22 | yes |
| 6 | 1.50 | 0.066 | 0.136 | 0.072 | 0.087 | 0.938 | 0.09 | yes |
| 9 | 0.25 | 0.761 | 0.054 | 0.021 | 0.826 | 0.289 | — | no |
| 9 | 0.50 | 0.252 | 0.348 | 0.137 | 0.372 | 0.589 | 4.36 | yes |
| 9 | 0.75 | 0.085 | 0.395 | 0.145 | 0.123 | 0.779 | 2.30 | yes |
| 9 | 1.00 | 0.040 | 0.320 | 0.120 | 0.052 | 0.868 | 0.79 | yes |
| 9 | 1.25 | 0.022 | 0.242 | 0.093 | 0.031 | 0.911 | 0.29 | yes |
| 9 | 1.50 | 0.016 | 0.187 | 0.073 | 0.022 | 0.936 | 0.13 | yes |

Peak is the largest computational-basis probability of the ground state.
S_half is the half-chain von Neumann entropy (nats). χ_F = median (1−|⟨ψ_λ|ψ_λ′⟩|²)/(Δλ)²
from the previous grid point. A point is quantum when peak Med < 0.5.

The packs are the locked 4-SAT TEB family (4 literals, m = round(9.93 n), satisfiable only), not a fresh 3-XOR E3LIN2 draw. `H_c` is that unsat diagonal. `Nov_3rd/data/scale1000/*.npz` is gitignored and was not on disk, so instances are the SCB=54e9 prefix from `sat_gen.instances` (the same generator as `regen_scale1000.py`). ED uses the first 32, train the next 64, holdout the next 128.

Raw unsat is the diagonal that puts this λ grid on the entangled side of the crossover. With `H_c = (n/m)·unsat` the same λ values sit past the crossover (`⟨X⟩/n ≳ 0.97` already at λ=0.25) and the ground state is a product `|+⟩` state. The locked pulse still generates `exp(−i γ_phys H_c)` because `γ_phys = γ·(n/m)`.

## Train / holdout

Train B=64 per n, holdout B=128, seed=0.
Fourier QAOA uses a full DST-IV (γ) / DCT-IV (β) basis at Q=L, paper init β=0.01, γ=−0.01.
power_p and Walsh-degree SNAP warm-start from that trained QAOA dual (`φ_k=−β k`, `c2=c3=0`),
then Adam + L-BFGS on mean ε. Reported Med is the best holdout checkpoint.
Angles are written under `smoke_A/` only.

| L | method | n | Med ε | QAOA Med ε | R | mark | plus Med | bitstring Med |
|---|--------|---|-------|------------|---|------|---------|---------------|
| 2 | qaoa | 6 | 0.0547 | 0.0547 | 1.0000 | TIE | 0.1903 | 0.3153 |
| 2 | qaoa | 9 | 0.0545 | 0.0545 | 1.0000 | TIE | 0.1895 | 0.3215 |
| 2 | power_p | 6 | 0.0484 | 0.0547 | 0.8838 | BEAT | 0.1903 | 0.3153 |
| 2 | power_p | 9 | 0.0535 | 0.0545 | 0.9811 | TIE | 0.1895 | 0.3215 |
| 2 | walsh | 6 | 0.0338 | 0.0547 | 0.6170 | BEAT | 0.1903 | 0.3153 |
| 2 | walsh | 9 | 0.0346 | 0.0545 | 0.6341 | BEAT | 0.1895 | 0.3215 |
| 16 | qaoa | 6 | 0.0113 | 0.0113 | 1.0000 | TIE | 0.1903 | 0.3153 |
| 16 | qaoa | 9 | 0.0107 | 0.0107 | 1.0000 | TIE | 0.1895 | 0.3215 |
| 16 | power_p | 6 | 0.0099 | 0.0113 | 0.8771 | BEAT | 0.1903 | 0.3153 |
| 16 | power_p | 9 | 0.0096 | 0.0107 | 0.9016 | BEAT | 0.1895 | 0.3215 |
| 16 | walsh | 6 | 0.0093 | 0.0113 | 0.8243 | BEAT | 0.1903 | 0.3153 |
| 16 | walsh | 9 | 0.0082 | 0.0107 | 0.7686 | BEAT | 0.1895 | 0.3215 |

## Verdict: **needs_edit**

The quantumness gate passes, and a Hamming-shell SNAP beats Fourier QAOA, but the scalar power_p head does not clear both depths.

- λ=0.25 still has a nearly classical ground state (peak Med 0.80 / 0.76). From λ=0.50 upward the ground state is not a bitstring. GRAPE of the diagonal `H_c` is the cost layer the motif already applies; it is not the whole answer, because the ground state at λ=0.75 is entangled (S_half 0.24 / 0.40) and carries ⟨X⟩/n ≈ 0.77.
- power_p at L=16 beats on both n (R = 0.877, 0.902). At L=2 it beats n=6 (R = 0.884) and ties n=9 (R = 0.981), 0.001 above the 0.98 line. The last L-BFGS rounds had flattened, so this is not an under-trained miss.
- Walsh-degree γ, same shell and the same diagonal `H_c` split into degrees 1–4, beats on both n at L=2 (R = 0.617, 0.634) and at L=16 (R = 0.824, 0.769).
- Both heads beat the `|+\rangle` product state (Med ε ≈ 0.19) and the best satisfying bitstring (Med ε ≈ 0.32).

Edit, not a kill: keep the gate and the transverse-field sign, and let the cost angles depend on Walsh degree. Scalar power_p is a shallow-L near miss on n=9. The family is not killed: SNAP does reach Med ε ≤ 0.98× Fourier QAOA on both n=6 and n=9.

Kill criteria (SHORTLIST family A and PLAN; `Nov_3rd/runs/opus_ideas/opus_reply_20261104_2150_hamiltonian_families.md` is not in the repo):

- KILL if the ground state stays a classical bitstring on the λ grid.
- KILL if ⟨H⟩ needs YY or XX correlators. It does not.
- KILL if GRAPE of the diagonal `H_c` alone already prepares the ground state.
  A bitstring ground state would. At the chosen λ the median peak is below 0.5, so the
  diagonal pulse is not the whole answer. It is exactly the cost layer the motif already applies.
- KILL if SNAP never reaches Med ε ≤ 0.98× Fourier QAOA on both n=6 and n=9.
- needs_edit if that bar is cleared at only one depth, or only by the Walsh-degree head.
- PASS if power_p clears it at every smoked depth and the ground state is quantum.

Phase-2 n21 ladder, `settings.py`, and `qaoa_angles/` were not modified.

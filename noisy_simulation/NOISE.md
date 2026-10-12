# Noise model of the noisy device simulator

This folder simulates the input circuit (`input_circuit/circuit.json`) on a realistic version of the
hardware in `docs/joint_parity_gate.tex`: data transmons d and e, coupler transmon q, cavities A and B,
wired as the chain d–A–q–B–e. The simulator evolves a density matrix and replaces only the
`probs_from_x` step of the noiseless optimizer, so the encoding, XOR relabel, leakage handling and
Gibbs cost are unchanged.

## 1. Master equation

The physics is the Lindblad equation of the JP note (`eq:lindblad`), in the interaction picture of
`eq:Htilde` (the free ω terms are removed; this is exact because every term is diagonal):

```
H = -chi_qA (n_A + n_B) P_e^q  -  chi_dA n_A P_e^d  -  chi_eB n_B P_e^e
    - K/2 [n_A(n_A-1) + n_B(n_B-1)]  -  K_AB n_A n_B

drho/dt = -i[H, rho] + kappa D[a] rho + kappa D[b] rho
          + sum_{j in d,q,e} ( gamma D[sigma_-^j] rho + (gamma_phi / 2) D[sigma_z^j] rho )
```

All three transmons carry T1 and T2 noise. The coupler q is part of the state during every JP gate, so
coupler decay and dephasing in the middle of the gate produce whatever parity-phase error they
physically cause. No effective "parity dephasing" channel is put in by hand.

## 2. Noise strengths and where they come from

| quantity | value | source | role |
|---|---|---|---|
| T1,cav | 436 μs | Eickbusch *et al.* 2022, quoted in Dutta App. A eq. (42) | κ = 1/T1,cav = 2.29×10³ s⁻¹ (κ/2π ≈ 365 Hz) |
| τ_ECD | 0.65 μs per (R_d, R_e, ECD_dA ∥ ECD_eB) block | Dutta App. A eq. (50): 3.27 μs / 5 ECD blocks, the conservative experimental value | idle time of one ECD block |
| κτ_ECD | 1.49×10⁻³ | Dutta App. A, per-block value "κτ ∼ 1.5×10⁻³" | cavity loss per ECD block |
| χ_qA = χ_qB | 2π × 1 MHz (matched) | JP note, unit table | sets the JP wait χτ = π |
| τ_JP | 2 × π/χ_qA = 1 μs | JP note eq. for τ_JP | two waits; the four q pulses (~24 ns) are instantaneous |
| κτ_JP | 2.29×10⁻³ | derived | cavity loss per JP gate |
| T1,q | 50 μs (d, q, e) | Eickbusch 2022 (JP note bibliography) | γ = 2.0×10⁴ s⁻¹, γ/χ_qA = 3.18×10⁻³ |
| T2 | 30 μs (d, q, e) | Eickbusch 2022 | γ_φ = 1/T2 − 1/(2T1) = 2.33×10⁴ s⁻¹, γ_φ/χ_qA = 3.7×10⁻³ |
| χ_dA = χ_eB | 0 by default (option: 2π × 33 kHz) | JP note: echoed by X_d X_e at the gate midpoint | spectator phase during the waits |
| K, K_AB | 0 by default (option: 2π × 1–10 kHz) | JP note (Wang 2016 range) | Kerr phases during the waits |

**How Dutta chose the cavity number.** Dutta *et al.* (Appendix A, eqs. 36–53) model photon loss as the
amplitude-damping channel, apply it once after every ECD-plus-rotation block, and express its strength
as the dimensionless product κτ = τ/T1,cav. They give two timing estimates. The pulse-limited floor is
T_ECD = 2t_q + 4t_D = 224 ns (eqs. 43–44). The conservative experimental value is 0.65 μs per block,
read from the Eickbusch binomial-code preparations (eq. 50). The conservative value gives
κτ ≈ 1.5×10⁻³ per block. Their optimizer still resolves the answer at κτ = 10⁻² and degrades around
κτ ≈ 0.1 (their Fig. 14). This simulator uses the conservative 0.65 μs.

**What this simulator adds to Dutta's model.** Dutta dropped transmon noise. Here every transmon gets
the same T1 = 50 μs, T2 = 30 μs, and the JP gate is simulated as its pulse/wait sequence with the
coupler live in the state. The transmon terms turn out to be the larger error (Section 6).

`Device` in `device.py` stores these laboratory values in SI units. The code uses only the
dimensionless products rate × time. `noise_scale` (`--noise-scale` / `--kappa-tau-scale`) multiplies
κ, γ and γ_φ together: 1 is the device, 0 is the noiseless reference.

## 3. Kraus channels (exact for each dissipator)

Each dissipator, integrated alone over a time τ, has a closed-form finite-time channel:

* **Photon loss** κD[a] (Dutta eqs. 37–39):
  `K_j = sqrt((1 - e^{-κτ})^j / j!) e^{-κτ n/2} a^j`. The operators are stored in shift form (a weight
  vector per j) and applied by slicing, so no nf×nf matrix product is needed. With a jump cutoff j_max,
  K_0 is completed as `(I - Σ_{j≥1} K_j†K_j)^{1/2}` (Dutta eq. 39) so the channel stays trace
  preserving. The completion changes K_0 at order (κτ)^{j_max+1}.
* **Transmon** γD[σ₋] + (γ_φ/2)D[σ_z]: amplitude damping with p = 1 − e^{−γτ}, composed with phase
  damping with coherence factor e^{−γ_φτ}. The two commute. Populations relax as e^{−γτ}, and
  coherences decay as e^{−τ/T2}, where 1/T2 = γ/2 + γ_φ.

## 4. One circuit layer

```
R_d, R_e, ECD_dA, ECD_eB      ideal unitaries (same gate order as noiseless/ecd_kbit)
idle tau_ECD                  photon loss on A, B; T1/T2 on d, e
JP gate                       q added in |g>
                              Ry(+π/2)_q
                              wait (χτ = π)            all dissipators active, Strang-split
                              Ry(+π/2) S† Ry(−π/2)_q
                              wait (χτ = π)
                              Ry(−π/2)_q
                              q traced out
```

Each wait applies the diagonal Hamiltonian exactly. It is split with the Kraus channels of every
dissipator (photon loss on A and B, T1/T2 on q, d and e) as K(h/2) U(h) K(h) … U(h) K(h/2), with
h = τ_wait / n_slice. Without noise the sequence equals `jp_diag` exactly, which gives the phase
e^{iπ(n_A+n_B)} on q = |e⟩ and leaves q in |g⟩.

Tracing out q at the end of the gate corresponds to a coupler reset between gates. The next JP gate
starts with q in |g⟩ again. The reduced state of (d, e, A, B) is exact either way. The reset only
matters for whatever residual excitation q would otherwise carry into the next gate.

### Two equivalent JP implementations

* `direct`: q is added to ρ (dimension 2 × 4nf²) and the sequence runs as written. This path supports
  every option, including the spectator shifts χ_dA and χ_eB.
* `fast` (default): if χ_dA = χ_eB = 0, then d and e enter the gate only through their own T1/T2
  channels, which commute with everything else in the gate. Their channel is applied once for
  τ_ECD + τ_JP. The cavity part, traced over q, is then a fixed linear map of shift form:
  `ρ[n,m,n',m'] ← Σ_{j,k} C_jk[n,m,n',m'] ρ[n−j, m−k, n'−j, m'−k]`. The map C is built once per
  device by running the same q-sequence on every Fock matrix element, and it also absorbs the
  preceding τ_ECD photon loss. C keeps up to J = 3 total losses per cavity per layer, with the exact
  (uncompleted) K_0. The dropped probability (> 3 losses in about 1.65 μs) shows up as a trace deficit
  below 10⁻¹⁰, and `probs_from_x` renormalizes it away.

  The fast path matches `direct` to 10⁻¹³ when no jumps are dropped. Building C at nf = 24 takes
  about 12 s and its file is about 85 MB. The runner builds it once and shares it with all workers as
  a read-only memory-mapped `.npy` under `results/channel_cache/`.

## 5. Accuracy

Three things are approximate:

1. **Strang splitting inside each JP wait.** Every channel and the Hamiltonian are individually exact.
   Their product is accurate to second order in h. A single slice per wait is too coarse, because
   χτ = π is large: a coupler decay at time t freezes the conditional phase at χt, and one slice
   places all decays at t = 0 or t = τ. Comparison with QuTiP `mesolve` of the full five-mode master
   equation at nf = 4, on a random input state, measured by trace distance of the (d, e, A, B) state
   after one JP gate:

   | n_slice | 1 | 2 | 4 | **8 (default)** | 16 | 32 |
   |---|---|---|---|---|---|---|
   | splitting error | 2.3e-3 | 9.9e-4 | 1.1e-4 | **2.4e-5** | 5.8e-6 | 1.4e-6 |

   The total effect of the noise on that state is 4.1e-2, so the default resolves the noise to about
   0.06%. Almost all of the splitting error comes from the transmon terms. Photon loss alone gives
   below 4e-7 after q is traced out. `n_slice` affects only the one-time build of C, so the
   per-evaluation cost does not depend on it.
2. **Photon-loss truncation.** The fast path drops more than 3 losses per cavity per layer (below
   10⁻¹⁰). The idle channel on the direct path uses j_max = 2 with K_0 completion; the first dropped
   term is (κτ n_max)³/3!.
3. **Hardware not modeled.** Pulses (qubit rotations, ECD) are ideal unitaries, with all idle noise
   of an ECD block lumped into one channel after the block (Dutta's scheme). Thermal population,
   transmon leakage to |f⟩, measurement error, Kerr (off by default) and spectator χ (echoed by
   default) are left out. A pulse-level ECD with the data transmon live mid-gate is the natural next
   extension.

## 6. Error budget (what comes out)

Contributions per JP gate, from the rates alone:

* coupler T1 while q is in a superposition: about γτ_JP/2 = 1%
* coupler dephasing: about γ_φτ_JP = 2.3%
* data transmons d and e over τ_ECD + τ_JP = 1.65 μs: γτ = 3.3% and γ_φτ = 3.8% each, for any
  coherence they hold
* cavity loss: κτ ≈ 3.8×10⁻³ per layer, times the photon number

Measured on one random L = 4 parameter set at nf = 24 (fidelity to the ideal state):

| noise switched on | fidelity | infidelity per layer |
|---|---|---|
| everything | 0.759 | 6.0% |
| cavity loss only | 0.975 | 0.63% |
| transmon T1 only (γ, no γ_φ) | 0.890 | 2.7% |
| transmon pure dephasing only (γ_φ, no γ) | 0.872 | 3.2% |

Transmon dephasing and T1 dominate. Their per-layer infidelities add almost linearly to the total. Cavity loss is the smallest term at these rates, consistent
with Dutta's conclusion that κτ ≈ 10⁻³ per block is benign. Optimized circuits at small β hold fewer
photons, so their cavity term is smaller still.

## 7. Tests (`tests/test_noise.py`)

The tests are layered so that a failure points to the faulty piece:

* **Channels:**
  * a Fock state decays binomially;
  * a coherent state shrinks to α e^{−κτ/2} and stays pure;
  * the qubit channel matches its closed forms and the T2 definition;
  * each dissipator agrees with its own `mesolve`;
  * the channels compose as a semigroup;
  * truncation orders hold (K_0 completion, j_max = 2 vs 4).
* **Machinery:**
  * every axis-wise helper matches explicit Kronecker-product matrices in the 4-mode and 5-mode shapes;
  * the conditional ECD form equals the explicit ECD(β)R gate;
  * every channel keeps ρ Hermitian, unit trace and positive;
  * adding q and tracing it out round-trips;
  * the fast JP channel equals `direct`;
  * the channel disk cache works.
* **End to end:**
  * zero noise reproduces the noiseless ket simulator to 10⁻¹²;
  * the ideal JP sequence equals `jp_diag`;
  * the wait Hamiltonian diagonal matches the operator form of `eq:Htilde`;
  * the full JP sequence matches QuTiP `mesolve` of the full master equation with q, d, e, A, B at
    nf = 6 (trace distance below 10⁻⁴ and below 0.5% of the noise effect), with and without spectator
    χ and Kerr; second-order convergence in n_slice is checked;
  * the zero-noise runner reproduces a noiseless optimizer trial on the same seed.

Run them with `python -m pytest -q noisy_simulation/tests` (about 30 s).

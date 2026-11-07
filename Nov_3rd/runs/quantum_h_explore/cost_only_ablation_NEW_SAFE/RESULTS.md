# NEW_SAFE cost-only ablation — U_c^X / SNAP mixer

_UTC 2026-11-04 22:10:12_

## Question

On NEW_SAFE, does a diagonal cost pulse by itself reach the ground state, or does the Hamming-shell SNAP mixer have to sit in the circuit with the cost pulse?

## Hamiltonian and circuits

\[
H = \mathrm{diag}(E_{\mathrm{unsat}}) - 0.75 \sum_{i=1}^{n} X_i
\]

Motif (arm A): \(U_c^X = W \exp(-i \sum_{w=1}^{4} \gamma_{\ell,w} E_w) W\) with \(W = H^{\otimes 3}\) per cavity, \(E_w\) the degree-\(w\) Walsh piece of `cost_Cn`, then a shared Hamming-shell SNAP \(\varphi_k = c_1 k + c_2 k^2 + c_3 k^3\), \(k=0..3\), and a final \(W\) into the Fock/bitstring readout. Equal \(\gamma_w\) and \(c_2=c_3=0\), \(c_1=-\beta\) reproduces Fourier/free QAOA with the repo mixer \(\exp(-i\beta X/2)\) (checked, energy error \(<10^{-6}\)).

| arm | what is trained | what is frozen |
|-----|-----------------|----------------|
| A | Walsh \(\gamma_{\ell,w}\) and shell \(c_1,c_2,c_3\) | — |
| B | Walsh \(\gamma\) only | shell phases = 0 (identity SNAP) |
| C | shell \(c_1,c_2,c_3\) | \(\gamma = 0\) |
| D | Zhou Fourier modes of \((\beta,\gamma)\), \(q=L\) | standard QAOA, no SNAP |

## Why a diagonal pulse cannot win by itself

With identity SNAP the layer is \(U_c^X(\gamma_\ell)=W e^{-i\phi_\ell} W\). These factors commute and collapse, for any number of layers and any diagonal waveform \(\phi(z)\) (scalar \(\gamma\), Walsh grading, or a full GRAPE schedule of \(H_c\)), to

\[
|\psi\rangle = e^{-i\phi(Z)}\,|+\rangle^{\otimes n}.
\]

Z-populations stay uniform, so \(\langle E_{\mathrm{unsat}}\rangle=\mathrm{mean}(E)\). Each \(\langle X_i\rangle=\mathrm{mean}_z\cos(\phi(z)-\phi(z\oplus e_i))\le 1\), with equality on every qubit only for constant \(\phi\). Minimizing \(\langle H\rangle=\langle E\rangle-0.75\sum\langle X_i\rangle\) inside this family lands on \(|+\rangle\), energy \(\mathrm{mean}(E)-0.75\,n\). Arm C is the same state: \(\gamma=0\) returns the register to \(|0\rangle\) before each Z-diagonal SNAP, the shell multiplies only a global phase, and the final \(W\) reads out \(|+\rangle\).

The training runs are a numerical check of that bound. Every trained arm-B seed ends on the \(|+\rangle\) row: the summed diagonal waveform (layers add, and they commute) has standard deviation about \(10^{-5}\) on n12, even when individual layers are nonzero and cancel. Arm C keeps median \(\langle E_{\mathrm{unsat}}\rangle=\mathrm{mean}(E)\) and median \(\sum\langle X_i\rangle=n\) for every shell coefficient that was tried.

## Protocol

- L ∈ {2, 4}; seeds {0,1,2,3,4} plus one A warm-start from the val-best D angles.
- Train objective: mean ⟨H⟩ on n6[0:64] and n9[0:64] (equal average). Optimizer: Adam lr=0.05 then L-BFGS, same step budget for A and D.
- Model selection uses the median gap on a disjoint val slice (n6/n9 indices [64:112]). n12 never enters the loss or the selection.
- Frozen eval: `train_packs/train_n12.npz` indices [0:64] primary and [64:128] confirm (scale1000 n12 is not in this tree; these instances never enter the loss). Extra: scale1000 n6[0:64], seed-disjoint from the n6 train pack.
- Arm D uses \(q=L\) Zhou modes, which span every \((\beta,\gamma)\) schedule, so D is full-angle QAOA. The QAOA-dual point inside A (equal \(\gamma_w\), \(c_2=c_3=0\)) reproduces D's gaps to printed precision.
- Metric: Medε = median over instances of (⟨H⟩ − E0). Ratios r_B = Medε_A/Medε_B, and the same for C and D. **PASS** if r_B, r_C, and r_D are all ≤ 0.85 on primary n12. **KILL** if r_B ≥ 0.95. Otherwise **needs_edit**.
- E0: dense `eigh` at n≤9, Lanczos `eigsh` at n=12 (residual checked).
- n21 ladder, `settings.py` locks, and `qaoa_angles/` were not touched.

## Ground state

| slice | med E0 | med E(|+⟩) | med ε(|+⟩) | med ε(best bitstring) | med GS bitstring peak | frac sat |
|-------|--------|------------|------------|------------------------|-----------------------|----------|
| n6_val | -1.9695 | -0.7500 | 1.2195 | 1.9695 | 0.1935 | 1.0000 |
| n9_val | -2.8409 | -1.1875 | 1.6534 | 2.8409 | 0.0788 | 1.0000 |
| n6_ext | -1.8960 | -0.7500 | 1.1460 | 1.8960 | 0.2017 | 1.0000 |
| n12 | -3.9223 | -1.5625 | 2.3598 | 3.9223 | 0.0366 | 1.0000 |
| n12b | -3.8250 | -1.5625 | 2.2625 | 3.8250 | 0.0392 | 1.0000 |

Every eval instance is satisfiable, so the best computational-basis state has energy 0. The ground state is lower, and its largest bitstring weight is well below 1/2, so the ground state is a superposition.

## Results

Headline rows are the val-selected checkpoint of each arm. Medε is the median instance gap. Lower is closer to E0.

### L=2

| arm | seed | val Medε | n6 ext Medε | n12 Medε | n12b Medε | n12 ⟨D⟩ med | n12 ⟨ΣX⟩ med |
|-----|------|----------|-------------|----------|-----------|-------------|--------------|
| A | warm:stage2 | 0.2419 | 0.2180 | **0.4279** | 0.4212 | 4.3611 | 10.3905 |
| B | 1 | 1.4364 | 1.1460 | **2.3598** | 2.2625 | 7.4375 | 12.0000 |
| C | 0 | 1.4364 | 1.1460 | **2.3598** | 2.2625 | 7.4375 | 12.0000 |
| D | 0 | 0.4076 | 0.3541 | **0.7290** | 0.6443 | 4.6721 | 10.4697 |
| |+> | analytic | 1.4364 | 1.1460 | 2.3598 | 2.2625 |  |  |

Primary n12 ratios: r_B=0.1813, r_C=0.1813, r_D=0.5870. Confirm n12b: r_B=0.1861, r_C=0.1861, r_D=0.6537.

Median-over-seeds n12 Medε: A=0.4859, B=2.3598, C=2.3598, D=0.7290. Ratios of those medians: r_B=0.2059, r_C=0.2059, r_D=0.6666.

**L=2 verdict: PASS.** Full Walsh SNAP + U_c^X is closer to E0 than the diagonal pulse, the SNAP-only circuit, and Fourier QAOA. Selected A is `warm:stage2` (warm-full), γ spread 0.9519, mean |c2|=0.0978, mean |c3|=0.0159.

<details><summary>all seeds, n12 Medε</summary>

| arm | seed | kind | val Medε | n12 Medε | γ spread | mean |c2| | mean |c3| |
|-----|------|------|----------|----------|----------|-------------|-------------|
| A | dual-of-D | dual-of-D | 0.4076 | 0.7290 | 0.0000 | 0.0000 | 0.0000 |
| A | 0:stage1 | subspace | 0.4076 | 0.7290 | 0.0000 | 0.0000 | 0.0000 |
| A | 0:stage2 | full | 0.2420 | 0.4279 | 0.9519 | 0.0975 | 0.0158 |
| A | 1:stage1 | subspace | 0.4318 | 0.7595 | 0.0000 | 0.0000 | 0.0000 |
| A | 1:stage2 | full | 0.2984 | 0.4859 | 0.7864 | 0.0166 | 0.0031 |
| A | 2:stage1 | subspace | 0.4318 | 0.7595 | 0.0000 | 0.0000 | 0.0000 |
| A | 2:stage2 | full | 0.2972 | 0.4886 | 0.9463 | 0.0422 | 0.0214 |
| A | 3:stage1 | subspace | 0.4076 | 0.7290 | 0.0000 | 0.0000 | 0.0000 |
| A | 3:stage2 | full | 0.2420 | 0.4279 | 0.9520 | 0.0967 | 0.0156 |
| A | 4:stage1 | subspace | 0.4318 | 0.7595 | 0.0000 | 0.0000 | 0.0000 |
| A | 4:stage2 | full | 0.2984 | 0.4860 | 0.8102 | 0.0207 | 0.0033 |
| A | warm:stage1 | warm-subspace | 0.4076 | 0.7290 | 0.0000 | 0.0000 | 0.0000 |
| A | warm:stage2 | warm-full | 0.2419 | 0.4279 | 0.9519 | 0.0978 | 0.0159 |
| B | 0 | identity_snap | 1.4364 | 2.3598 | 0.0000 | 0.0000 | 0.0000 |
| B | 1 | identity_snap | 1.4364 | 2.3598 | 0.0907 | 0.0000 | 0.0000 |
| B | 2 | identity_snap | 1.4364 | 2.3598 | 0.2216 | 0.0000 | 0.0000 |
| B | 3 | identity_snap | 1.4364 | 2.3598 | 0.1675 | 0.0000 | 0.0000 |
| B | 4 | identity_snap | 1.4364 | 2.3598 | 0.1952 | 0.0000 | 0.0000 |
| C | 0 | snap_only | 1.4364 | 2.3598 | 0.0000 | 0.0000 | 0.0000 |
| C | 1 | snap_only | 1.4364 | 2.3598 | 0.0000 | 0.2142 | 0.0379 |
| C | 2 | snap_only | 1.4364 | 2.3598 | 0.0000 | 0.2219 | 0.0442 |
| C | 3 | snap_only | 1.4364 | 2.3598 | 0.0000 | 0.1518 | 0.0222 |
| C | 4 | snap_only | 1.4364 | 2.3598 | 0.0000 | 0.1633 | 0.0566 |
| D | 0 | fourier | 0.4076 | 0.7290 | — | — | — |
| D | 1 | fourier | 0.4076 | 0.7290 | — | — | — |
| D | 2 | fourier | 0.5726 | 1.0005 | — | — | — |
| D | 3 | fourier | 0.4076 | 0.7290 | — | — | — |
| D | 4 | fourier | 0.4076 | 0.7290 | — | — | — |

</details>

### L=4

| arm | seed | val Medε | n6 ext Medε | n12 Medε | n12b Medε | n12 ⟨D⟩ med | n12 ⟨ΣX⟩ med |
|-----|------|----------|-------------|----------|-----------|-------------|--------------|
| A | 4:stage2 | 0.1013 | 0.0919 | **0.1808** | 0.1869 | 3.6779 | 9.9310 |
| B | 1 | 1.4364 | 1.1460 | **2.3598** | 2.2625 | 7.4375 | 12.0000 |
| C | 0 | 1.4364 | 1.1460 | **2.3598** | 2.2625 | 7.4375 | 12.0000 |
| D | 0 | 0.1721 | 0.1548 | **0.3202** | 0.3153 | 3.9385 | 10.0138 |
| |+> | analytic | 1.4364 | 1.1460 | 2.3598 | 2.2625 |  |  |

Primary n12 ratios: r_B=0.0766, r_C=0.0766, r_D=0.5646. Confirm n12b: r_B=0.0826, r_C=0.0826, r_D=0.5927.

Median-over-seeds n12 Medε: A=0.1816, B=2.3598, C=2.3598, D=0.3566. Ratios of those medians: r_B=0.0769, r_C=0.0769, r_D=0.5092.

**L=4 verdict: PASS.** Full Walsh SNAP + U_c^X is closer to E0 than the diagonal pulse, the SNAP-only circuit, and Fourier QAOA. Selected A is `4:stage2` (full), γ spread 0.7978, mean |c2|=0.0976, mean |c3|=0.0180.

<details><summary>all seeds, n12 Medε</summary>

| arm | seed | kind | val Medε | n12 Medε | γ spread | mean |c2| | mean |c3| |
|-----|------|------|----------|----------|----------|-------------|-------------|
| A | dual-of-D | dual-of-D | 0.1721 | 0.3202 | 0.0000 | 0.0000 | 0.0000 |
| A | 0:stage1 | subspace | 0.1721 | 0.3202 | 0.0000 | 0.0000 | 0.0000 |
| A | 0:stage2 | full | 0.1014 | 0.1809 | 0.7890 | 0.1033 | 0.0191 |
| A | 1:stage1 | subspace | 0.2080 | 0.3566 | 0.0000 | 0.0000 | 0.0000 |
| A | 1:stage2 | full | 0.1332 | 0.2209 | 0.8667 | 0.0929 | 0.0177 |
| A | 2:stage1 | subspace | 0.2070 | 0.3838 | 0.0000 | 0.0000 | 0.0000 |
| A | 2:stage2 | full | 0.1415 | 0.2356 | 1.2763 | 0.2153 | 0.0558 |
| A | 3:stage1 | subspace | 0.1721 | 0.3201 | 0.0000 | 0.0000 | 0.0000 |
| A | 3:stage2 | full | 0.1018 | 0.1812 | 0.7803 | 0.1111 | 0.0213 |
| A | 4:stage1 | subspace | 0.1721 | 0.3202 | 0.0000 | 0.0000 | 0.0000 |
| A | 4:stage2 | full | 0.1013 | 0.1808 | 0.7978 | 0.0976 | 0.0180 |
| A | warm:stage1 | warm-subspace | 0.1721 | 0.3202 | 0.0000 | 0.0000 | 0.0000 |
| A | warm:stage2 | warm-full | 0.1017 | 0.1816 | 0.7755 | 0.1178 | 0.0228 |
| B | 0 | identity_snap | 1.4364 | 2.3598 | 0.0000 | 0.0000 | 0.0000 |
| B | 1 | identity_snap | 1.4364 | 2.3598 | 0.2062 | 0.0000 | 0.0000 |
| B | 2 | identity_snap | 1.4364 | 2.3598 | 0.2453 | 0.0000 | 0.0000 |
| B | 3 | identity_snap | 1.4364 | 2.3598 | 0.2409 | 0.0000 | 0.0000 |
| B | 4 | identity_snap | 1.4364 | 2.3598 | 0.1604 | 0.0000 | 0.0000 |
| C | 0 | snap_only | 1.4364 | 2.3598 | 0.0000 | 0.0000 | 0.0000 |
| C | 1 | snap_only | 1.4364 | 2.3598 | 0.0000 | 0.1733 | 0.0476 |
| C | 2 | snap_only | 1.4364 | 2.3598 | 0.0000 | 0.1117 | 0.0149 |
| C | 3 | snap_only | 1.4364 | 2.3598 | 0.0000 | 0.1891 | 0.0575 |
| C | 4 | snap_only | 1.4364 | 2.3598 | 0.0000 | 0.0801 | 0.0274 |
| D | 0 | fourier | 0.1721 | 0.3202 | — | — | — |
| D | 1 | fourier | 0.2080 | 0.3566 | — | — | — |
| D | 2 | fourier | 0.3254 | 0.5950 | — | — | — |
| D | 3 | fourier | 0.2651 | 0.4827 | — | — | — |
| D | 4 | fourier | 0.2080 | 0.3566 | — | — | — |

</details>

## Verdict

**PASS**

L=2 n12 Medε A/B/C/D = 0.4279/2.3598/2.3598/0.7290 (r_B=0.181, r_C=0.181, r_D=0.587, verdict PASS). L=4 n12 Medε A/B/C/D = 0.1808/2.3598/2.3598/0.3202 (r_B=0.077, r_C=0.077, r_D=0.565, verdict PASS). Confirm-slice ratios stay in the same range (L=2 r_D=0.654, L=4 r_D=0.593), and the median over seeds does too (L=2 r_D=0.667, L=4 r_D=0.509).

Val-selected Walsh SNAP + \(U_c^X\) cuts the frozen-n12 median gap to \(E_0\) by more than the pre-registered 15% versus the identity-SNAP cost pulse, versus SNAP with the cost pulse removed, and versus same-depth Fourier QAOA, at both depths. A GRAPE schedule of diagonal \(H_c\) is the arm-B family. Its best state is \(|+\rangle\), median gap 2.36 at n=12. The interleaved mixer reaches 0.43 (L=2) and 0.18 (L=4).

The selected L=2 state does that by moving both observables: median \(\langle E_{\mathrm{unsat}}\rangle\) falls from 7.44 (\(|+\rangle\)) to 4.36, while median \(\sum\langle X_i\rangle\) falls from 12 to 10.39. Fourier QAOA reaches 4.67 and 10.47. Selected A leaves the QAOA subspace (\(\gamma\) spread \(\approx 0.8\)–\(0.95\), mean \(|c_2|\approx 0.10\)). The dual-of-D checkpoint inside A matches D exactly, so the extra gap is the Walsh grading plus the nonlinear Hamming shell, not a second optimization of the same QAOA angles. The cost pulse alone is not the ground state; the SNAP mixer has to be in the circuit with it.

## Repro

```bash
python3 Nov_3rd/scripts/cost_only_ablation_NEW_SAFE.py
```

Self-check (dual energy error 5.77e-15, ansatz TV 2.68e-07 / 1.60e-07) runs at startup. Wall 190s. Machine-readable: `results.json`.


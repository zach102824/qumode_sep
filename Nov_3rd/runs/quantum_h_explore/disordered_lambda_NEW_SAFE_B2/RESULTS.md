# Disordered λ at NEW_SAFE, B=2 Walsh SNAP (μ = 0)

**Overall: PASS.**

Same disorder as PR #49 (one pattern, seed 20261105, δ ∈ {0.15, 0.30}, μ = 0). The SNAP arm is the B=2 parameterization from PR #57: two coefficients on popcount, φ_k = c1 k + c2 k², not the four-phase Hamming shell. Fourier QAOA is matched. Λ is the ground state and does not depend on B.

## Hamiltonian

\[
H = \mathrm{diag}(\mathrm{unsat}) - \sum_i \lambda_i X_i
\]

\[
\lambda_i = 0.75\,(1 + \delta u_i),\qquad u_i\sim\mathrm{Unif}[-1,1]
\]

μ = 0. One pattern u (NumPy seed 20261105), shared by both strengths and by every n (qubit i uses u_i, so the n=6 pattern is the prefix of n=12). All λ_i stay positive. Readout is the Z histogram (unsat) plus the X histogram. The field is \(\sum_i \lambda_i\langle X_i\rangle\). No Y, no XX.

Λ_c = ⟨(Σ_{i∈c} X_i / 3)²⟩ from the X histogram of the true ground state. The reported median pools cavities and instances. Clean NEW_SAFE (δ = 0) has n=12 median Λ = 0.7477.

## Ansatz

| method | angles / layer | unitary |
|---|---|---|
| Fourier QAOA | 2: γ, β | \(U_X(\beta)\,U_C(\gamma)\), same β on every qubit |
| B=2 Walsh SNAP | 6: γ_{w=1..4}, c1, c2 | Walsh degree-1..4 cost, then φ_k = c1 k + c2 k² on every cavity |

B=2 coordinates are an orthonormal basis of span{k, k²} on k = 0..3, the same metric the locked shell uses on φ. A constant added to every popcount is a global phase and is not a coefficient. The paper init φ_k = −β k sits in this span exactly (reconstruction error 3.469e-18). Paper init β=0.01, γ=γ_w=-0.01, so Fourier and B=2 start on the same state. No per-qubit angles were added. Disorder is only in H. Cost family is still \(C_n=\mathrm{unsat}\cdot n/m\).

The PR #49 arm was the Hamming shell, four free phases φ_0..φ_3 (8 angles per layer with the four γ weights). This run replaces only that mixer budget. PR #57 found B=2 matches that shell at clean NEW_SAFE, L=2 (R = 0.6197, Medε 0.4261 / 0.6876).

## Protocol

- Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Checkpoint on the n=9 holdout [200:250). Reported n=6 and n=9 numbers are that holdout. Freeze n=12, first 128 instances.
- Seeds {0, 1, 2, 3, 4}. The seed only reshuffles batches. Every δ, method, and depth shares that stream (NumPy Generator(seed), the PR #49 stream).
- L-BFGS matches `Nov_3rd.settings`: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n=6 / half n=9). New optimizer each round. Loss = mean ε/n. No Adam.
- Each seed restores the round with the best n=9-holdout Med ε.
- R_Fourier = median over seeds of Medε_B2, divided by the same median for Fourier.
- n=21 was not loaded. `settings.py`, `qaoa_angles/`, and the phase-2 ladder were not written.

## Gate

- **PASS** a δ when frozen-n12 L=2 R_Fourier ≤ 0.85 and the n=12 median cavity Λ ≤ 0.80. That is B=2 keeping the disorder lead.
- **needs_edit** when B=2 loses the disorder lead: L=2 R is soft, in (0.85, 0.98], or median Λ sits in (0.80, 0.90).
- **KILL** when L=2 R_Fourier > 0.98 or median Λ ≥ 0.90. A KILL clause wins.
- Both δ=0.15 and δ=0.30 have to pass. One KILL makes the run KILL. L=4 is reported and is not the gate. The shell's disordered L=2 ratios were 0.613 and 0.617. Clean B=2 L=2 was 0.620.

## References (not remeasured)

δ = 0 is clean NEW_SAFE, B=2, from PR #57. The shell columns are PR #49 on this same disorder.

| source | δ | L | Medε Fourier | Medε SNAP | R_Fourier | n12 median Λ |
|---|---:|---:|---:|---:|---:|---:|
| PR #57 B=2 | 0 | 2 | 0.6876 | 0.4261 | 0.6197 | 0.7477 |
| PR #57 B=2 | 0 | 4 | 0.3172 | 0.1958 | 0.6174 | 0.7477 |
| PR #49 shell | 0.15 | 2 | 0.6772 | 0.4151 | 0.6130 | 0.7586 |
| PR #49 shell | 0.15 | 4 | 0.3125 | 0.2578 | 0.8249 | 0.7586 |
| PR #49 shell | 0.30 | 2 | 0.6694 | 0.4128 | 0.6167 | 0.7610 |
| PR #49 shell | 0.30 | 4 | 0.3130 | 0.3036 | 0.9699 | 0.7610 |

## λ patterns

### δ=0.15 (d0p15)

n=12 λ = 0.8090, 0.8535, 0.8490, 0.7594, 0.7723, 0.6785, 0.6494, 0.6893, 0.8033, 0.8295, 0.7300, 0.8291.

| cavity | λ | range |
|---|---|---:|
| 0 | 0.8090, 0.8535, 0.8490 | 0.0445 |
| 1 | 0.7594, 0.7723, 0.6785 | 0.0938 |
| 2 | 0.6494, 0.6893, 0.8033 | 0.1538 |
| 3 | 0.8295, 0.7300, 0.8291 | 0.0994 |

### δ=0.30 (d0p30)

n=12 λ = 0.8680, 0.9570, 0.9480, 0.7687, 0.7947, 0.6071, 0.5489, 0.6285, 0.8565, 0.9090, 0.7101, 0.9081.

| cavity | λ | range |
|---|---|---:|
| 0 | 0.8680, 0.9570, 0.9480 | 0.0889 |
| 1 | 0.7687, 0.7947, 0.6071 | 0.1876 |
| 2 | 0.5489, 0.6285, 0.8565 | 0.3077 |
| 3 | 0.9090, 0.7101, 0.9081 | 0.1989 |

## Ground state

Median Λ pools every cavity of every instance. n=6 and n=9 use the full pack (B=250). n=12 is the freeze pool (B=128). Λ does not depend on the SNAP budget, so these rows should sit on the PR #49 values.

| δ | n | B | median Λ | min Λ | max Λ | frac Λ>0.80 | frac Λ≥0.90 | per-cavity median | median Z-peak | frac peak≥0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0.15 | 6 | 250 | 0.7796 | 0.5701 | 0.9412 | 0.384 | 0.018 | 0.7852, 0.7697 | 0.1823 | 0.000 |
| 0.15 | 9 | 250 | 0.7603 | 0.5507 | 0.9348 | 0.267 | 0.009 | 0.7735, 0.7479, 0.7564 | 0.0870 | 0.000 |
| 0.15 | 12 | 128 | 0.7586 | 0.5590 | 0.9143 | 0.268 | 0.006 | 0.7884, 0.7444, 0.7413, 0.7686 | 0.0334 | 0.000 |
| 0.30 | 6 | 250 | 0.7956 | 0.5642 | 0.9439 | 0.480 | 0.022 | 0.8126, 0.7725 | 0.1681 | 0.000 |
| 0.30 | 9 | 250 | 0.7604 | 0.5578 | 0.9341 | 0.288 | 0.015 | 0.7922, 0.7464, 0.7395 | 0.0843 | 0.000 |
| 0.30 | 12 | 128 | 0.7610 | 0.5529 | 0.9166 | 0.314 | 0.021 | 0.8122, 0.7408, 0.7271, 0.7815 | 0.0312 | 0.000 |

Clean n=12 median Λ was 0.7477. PR #49 reported disordered n=12 median Λ 0.7586 and 0.7610.

## Ratios

Holdout at n=6 and n=9 is the checkpoint distribution (B=50 per seed). The decision row is n=12, L=2. Medε is the median over seeds of each seed's median excess.

| δ | L | n | Medε Fourier | Medε B=2 | R_Fourier | paired B2<Fourier | seeds ≤0.85 | #49 shell R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 2 | 6 | 0.3368 | 0.2031 | 0.6031 | 250/250 |  |  |
| 0.15 | 2 | 9 | 0.4652 | 0.2844 | 0.6115 | 250/250 |  |  |
| 0.15 | 2 | 12 | 0.6772 | 0.4143 | 0.6118 | 640/640 | 5/5 | 0.6130 |
| 0.15 | 4 | 6 | 0.1368 | 0.0863 | 0.6309 | 250/250 |  |  |
| 0.15 | 4 | 9 | 0.2310 | 0.1298 | 0.5620 | 250/250 |  |  |
| 0.15 | 4 | 12 | 0.3125 | 0.1885 | 0.6032 | 640/640 | 5/5 | 0.8249 |
| 0.30 | 2 | 6 | 0.3156 | 0.1770 | 0.5610 | 250/250 |  |  |
| 0.30 | 2 | 9 | 0.4737 | 0.2978 | 0.6287 | 250/250 |  |  |
| 0.30 | 2 | 12 | 0.6694 | 0.4121 | 0.6156 | 640/640 | 5/5 | 0.6167 |
| 0.30 | 4 | 6 | 0.1324 | 0.0827 | 0.6247 | 244/250 |  |  |
| 0.30 | 4 | 9 | 0.2291 | 0.1451 | 0.6334 | 250/250 |  |  |
| 0.30 | 4 | 12 | 0.3130 | 0.2003 | 0.6398 | 640/640 | 5/5 | 0.9699 |

### Seeds at n=12

| δ | L | seed | Medε B=2 | Medε Fourier | R | shell RMS | phys RMS | best round B=2 | wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 2 | 0 | 0.4132 | 0.6772 | 0.6101 | 0.0524 | 0.00 | 38 | 4.5 |
| 0.15 | 2 | 1 | 0.4152 | 0.6771 | 0.6133 | 0.0573 | 0.00 | 16 | 4.5 |
| 0.15 | 2 | 2 | 0.4144 | 0.6772 | 0.6120 | 0.0519 | 0.00 | 18 | 4.6 |
| 0.15 | 2 | 3 | 0.4143 | 0.6771 | 0.6119 | 0.0536 | 0.00 | 24 | 4.6 |
| 0.15 | 2 | 4 | 0.4137 | 0.6776 | 0.6105 | 0.0549 | 0.00 | 16 | 4.6 |
| 0.15 | 4 | 0 | 0.1893 | 0.3128 | 0.6052 | 0.1039 | 0.00 | 38 | 6.9 |
| 0.15 | 4 | 1 | 0.1899 | 0.3134 | 0.6060 | 0.1042 | 0.00 | 39 | 6.4 |
| 0.15 | 4 | 2 | 0.1885 | 0.3125 | 0.6032 | 0.1142 | 0.00 | 38 | 6.5 |
| 0.15 | 4 | 3 | 0.1864 | 0.3101 | 0.6012 | 0.1020 | 0.00 | 39 | 6.1 |
| 0.15 | 4 | 4 | 0.1869 | 0.3124 | 0.5984 | 0.0926 | 0.00 | 39 | 6.1 |
| 0.30 | 2 | 0 | 0.4109 | 0.6694 | 0.6138 | 0.0581 | 0.00 | 21 | 4.5 |
| 0.30 | 2 | 1 | 0.4118 | 0.6679 | 0.6165 | 0.0727 | 0.00 | 11 | 4.8 |
| 0.30 | 2 | 2 | 0.4128 | 0.6687 | 0.6173 | 0.0625 | 0.00 | 16 | 4.6 |
| 0.30 | 2 | 3 | 0.4121 | 0.6696 | 0.6155 | 0.0547 | 0.00 | 26 | 4.5 |
| 0.30 | 2 | 4 | 0.4141 | 0.6710 | 0.6172 | 0.0574 | 0.00 | 34 | 4.5 |
| 0.30 | 4 | 0 | 0.2003 | 0.3147 | 0.6363 | 0.1070 | 0.00 | 38 | 6.6 |
| 0.30 | 4 | 1 | 0.2017 | 0.3119 | 0.6466 | 0.1088 | 0.00 | 39 | 6.5 |
| 0.30 | 4 | 2 | 0.2011 | 0.3123 | 0.6439 | 0.1115 | 0.00 | 39 | 6.4 |
| 0.30 | 4 | 3 | 0.1988 | 0.3130 | 0.6351 | 0.1060 | 0.00 | 38 | 6.6 |
| 0.30 | 4 | 4 | 0.1992 | 0.3130 | 0.6366 | 0.1032 | 0.00 | 39 | 6.5 |

## Calls

| δ | call | L=2 R | n12 median Λ | #49 shell L=2 R | detail |
|---:|---|---:|---:|---:|---|
| 0.15 | PASS | 0.6118 | 0.7586 | 0.6130 | L=2 R_Fourier ≤ 0.85 and median Λ ≤ 0.80; B=2 keeps the disorder lead |
| 0.30 | PASS | 0.6156 | 0.7610 | 0.6167 | L=2 R_Fourier ≤ 0.85 and median Λ ≤ 0.80; B=2 keeps the disorder lead |

## Reading

δ=0.15: n12 median Λ=0.7586 (PR #49 0.7586, clean 0.7477, shift from clean +0.0109). Per-cavity medians 0.7884, 0.7444, 0.7413, 0.7686. L=2 Medε B=2=0.4143, Fourier=0.6772, R_Fourier=0.6118 (shell R 0.6130, shell Medε 0.4151; clean B=2 R 0.6197). 5/5 seeds ≤ 0.85. L=4 R_Fourier=0.6032 (shell L=4 R 0.8249). 26.8% of n=12 cavity-instance pairs sit above 0.80, 0.6% at or above 0.90. Median Z-peak=0.0334. B=2 keeps the disorder lead. Call **PASS**.

δ=0.30: n12 median Λ=0.7610 (PR #49 0.7610, clean 0.7477, shift from clean +0.0133). Per-cavity medians 0.8122, 0.7408, 0.7271, 0.7815. L=2 Medε B=2=0.4121, Fourier=0.6694, R_Fourier=0.6156 (shell R 0.6167, shell Medε 0.4128; clean B=2 R 0.6197). 5/5 seeds ≤ 0.85. L=4 R_Fourier=0.6398 (shell L=4 R 0.9699). 31.4% of n=12 cavity-instance pairs sit above 0.80, 2.1% at or above 0.90. Median Z-peak=0.0312. B=2 keeps the disorder lead. Call **PASS**.

B=2 keeps the disorder lead at both strengths. The L=2 Fourier ratio stays at or under 0.85, on the same side of the bar as the Hamming shell (0.613 / 0.617) and the clean B=2 cell (0.620). The n=12 median cavity Λ stays at or under 0.80. Two SNAP coefficients are enough for the disordered transverse field that four phases handled.

L=4 is reported and is not the gate. Clean B=2 at L=4 was a win (R=0.617), while the disordered shell was softer (0.825 / 0.970). δ=0.15 n12 R=0.603 (5/5 seeds ≤ 0.85), n6 holdout R=0.631. δ=0.30 n12 R=0.640 (5/5 seeds ≤ 0.85), n6 holdout R=0.625.

B=2 shells left the linear X mixer (δ=0.15 L=2 shell RMS 0.0536 (max phys RMS 0.00); δ=0.30 L=2 shell RMS 0.0581 (max phys RMS 0.00)). Shell RMS is the distance of φ_k from the best line a + b k, which for this family is the k² content. Phys RMS is the distance outside span{1, k, k²} and is zero when the parameterization holds. Zero shell RMS would be φ_k = b k.

Per-seed JSON is `{d0p15,d0p30}_L{L}_{method}_seed{s}.json`. Ground-state arrays are `gs_{tag}_n{n}_{start}_{stop}.npz`. `lambda_u.json` is the shared pattern. `summary.json` is the table source.

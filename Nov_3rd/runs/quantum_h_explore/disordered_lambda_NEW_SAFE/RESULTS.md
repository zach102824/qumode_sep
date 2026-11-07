# Disordered λ at NEW_SAFE (μ = 0)

**Overall: PASS.**

Same disorder as the shallow HVA run (one pattern, seed 20261105, δ ∈ {0.15, 0.30}), moved onto the Λ-safe transverse-field point. There is no cavity XX. Walsh Hamming-shell SNAP is compared with Fourier QAOA.

## Hamiltonian

\[
H = \mathrm{diag}(\mathrm{unsat}) - \sum_i \lambda_i X_i
\]

\[
\lambda_i = 0.75\,(1 + \delta u_i),\qquad u_i\sim\mathrm{Unif}[-1,1]
\]

μ = 0. One pattern u (NumPy seed 20261105), shared by both strengths and by every n (qubit i uses u_i, so the n=6 pattern is the prefix of n=12). All λ_i stay positive. Readout is the Z histogram (unsat) plus the X histogram. The field is \(\sum_i \lambda_i\langle X_i\rangle\). No Y, no XX.

Λ_c = ⟨(Σ_{i∈c} X_i / 3)²⟩ from the X histogram of the true ground state. The reported median pools cavities and instances. The uniform reference is n=12 median Λ = 0.7477 at λ = 0.75.

## Ansatz

| method | angles / layer | unitary |
|---|---|---|
| Fourier QAOA | 2: γ, β | \(U_X(\beta)\,U_C(\gamma)\), same β on every qubit |
| SNAP Walsh + Hamming shell | 8: γ_{w=1..4}, φ_{k=0..3} | Walsh degree-1..4 cost, then one shared shell on every cavity |

No per-qubit angles were added. Disorder is only in H. Cost family is still \(C_n=\mathrm{unsat}\cdot n/m\). Paper init β=0.01, γ=γ_w=-0.01, φ_k=−β k, so both methods start on the same state. A Fourier mixer has shell RMS 0.

## Protocol

- Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Checkpoint on the n=9 holdout [200:250). Reported n=6 and n=9 numbers are that holdout. Freeze n=12, first 128 instances.
- Seeds {0, 1, 2, 3, 4}. The seed only reshuffles batches. Every δ, method, and depth shares that stream.
- L-BFGS matches `Nov_3rd.settings`: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n=6 / half n=9). New optimizer each round. Loss = mean ε/n. No Adam.
- Each seed restores the round with the best n=9-holdout Med ε.
- R_Fourier = median over seeds of Medε_SNAP, divided by the same median for Fourier.
- n=21 was not loaded. `settings.py`, `qaoa_angles/`, and the phase-2 ladder were not written.

## Gate

- **PASS** a δ when frozen-n12 L=2 R_Fourier ≤ 0.85 and the n=12 median cavity Λ ≤ 0.80.
- **needs_edit** when L=2 R is soft, in (0.85, 0.98], or median Λ sits in (0.80, 0.90).
- **KILL** when L=2 R_Fourier > 0.98 or median Λ ≥ 0.90. A KILL clause wins.
- Both δ=0.15 and δ=0.30 have to pass. One KILL makes the run KILL. L=4 is reported and is not the gate. The uniform L=2 ratio is 0.620.

## Uniform reference (not remeasured)

δ=0 is bakeoff cell C (`bakeoff_Lambda_safe_top3`), same packs and budget, μ=0, λ=0.75.

| L | Medε Fourier | Medε SNAP | R_Fourier | n12 median Λ |
|---:|---:|---:|---:|---:|
| 2 | 0.6876 | 0.4264 | 0.6201 | 0.7477 |
| 4 | 0.3172 | 0.2755 | 0.8687 | 0.7477 |

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

Median Λ pools every cavity of every instance. n=6 and n=9 use the full pack (B=250). n=12 is the freeze pool (B=128).

| δ | n | B | median Λ | min Λ | max Λ | frac Λ>0.80 | frac Λ≥0.90 | per-cavity median | median Z-peak | frac peak≥0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0.15 | 6 | 250 | 0.7796 | 0.5701 | 0.9412 | 0.384 | 0.018 | 0.7852, 0.7697 | 0.1823 | 0.000 |
| 0.15 | 9 | 250 | 0.7603 | 0.5507 | 0.9348 | 0.267 | 0.009 | 0.7735, 0.7479, 0.7564 | 0.0870 | 0.000 |
| 0.15 | 12 | 128 | 0.7586 | 0.5590 | 0.9143 | 0.268 | 0.006 | 0.7884, 0.7444, 0.7413, 0.7686 | 0.0334 | 0.000 |
| 0.30 | 6 | 250 | 0.7956 | 0.5642 | 0.9439 | 0.480 | 0.022 | 0.8126, 0.7725 | 0.1681 | 0.000 |
| 0.30 | 9 | 250 | 0.7604 | 0.5578 | 0.9341 | 0.288 | 0.015 | 0.7922, 0.7464, 0.7395 | 0.0843 | 0.000 |
| 0.30 | 12 | 128 | 0.7610 | 0.5529 | 0.9166 | 0.314 | 0.021 | 0.8122, 0.7408, 0.7271, 0.7815 | 0.0312 | 0.000 |

Uniform n=12 median Λ was 0.7477, median Z-peak 0.0379.

## Ratios

Holdout at n=6 and n=9 is the checkpoint distribution (B=50 per seed). The decision row is n=12, L=2.

| δ | L | n | Medε Fourier | Medε SNAP | R_Fourier | paired SNAP<Fourier | seeds ≤0.85 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 2 | 6 | 0.3368 | 0.2015 | 0.5982 | 250/250 |  |
| 0.15 | 2 | 9 | 0.4652 | 0.2874 | 0.6180 | 250/250 |  |
| 0.15 | 2 | 12 | 0.6772 | 0.4151 | 0.6130 | 640/640 | 5/5 |
| 0.15 | 4 | 6 | 0.1368 | 0.1176 | 0.8590 | 195/250 |  |
| 0.15 | 4 | 9 | 0.2310 | 0.1953 | 0.8456 | 236/250 |  |
| 0.15 | 4 | 12 | 0.3125 | 0.2578 | 0.8249 | 622/640 | 3/5 |
| 0.30 | 2 | 6 | 0.3156 | 0.1763 | 0.5587 | 250/250 |  |
| 0.30 | 2 | 9 | 0.4737 | 0.3009 | 0.6352 | 250/250 |  |
| 0.30 | 2 | 12 | 0.6694 | 0.4128 | 0.6167 | 640/640 | 5/5 |
| 0.30 | 4 | 6 | 0.1324 | 0.1314 | 0.9923 | 151/250 |  |
| 0.30 | 4 | 9 | 0.2291 | 0.2153 | 0.9397 | 188/250 |  |
| 0.30 | 4 | 12 | 0.3130 | 0.3036 | 0.9699 | 513/640 | 1/5 |

### Seeds at n=12

| δ | L | seed | SNAP | Fourier | R | shell RMS | best round SNAP | wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 2 | 0 | 0.4169 | 0.6772 | 0.6156 | 0.1459 | 21 | 3.7 |
| 0.15 | 2 | 1 | 0.4100 | 0.6771 | 0.6056 | 0.0419 | 39 | 3.9 |
| 0.15 | 2 | 2 | 0.4151 | 0.6772 | 0.6130 | 0.1153 | 31 | 3.8 |
| 0.15 | 2 | 3 | 0.4140 | 0.6771 | 0.6114 | 0.0773 | 26 | 3.8 |
| 0.15 | 2 | 4 | 0.4157 | 0.6776 | 0.6135 | 0.0692 | 34 | 3.9 |
| 0.15 | 4 | 0 | 0.2552 | 0.3128 | 0.8158 | 0.2320 | 38 | 5.1 |
| 0.15 | 4 | 1 | 0.2751 | 0.3134 | 0.8777 | 0.2161 | 39 | 5.0 |
| 0.15 | 4 | 2 | 0.2578 | 0.3125 | 0.8249 | 0.2383 | 38 | 5.1 |
| 0.15 | 4 | 3 | 0.2835 | 0.3101 | 0.9143 | 0.2140 | 39 | 5.1 |
| 0.15 | 4 | 4 | 0.2446 | 0.3124 | 0.7831 | 0.2462 | 39 | 5.1 |
| 0.30 | 2 | 0 | 0.4097 | 0.6694 | 0.6119 | 0.1246 | 38 | 3.7 |
| 0.30 | 2 | 1 | 0.4128 | 0.6679 | 0.6180 | 0.1407 | 17 | 3.8 |
| 0.30 | 2 | 2 | 0.4218 | 0.6687 | 0.6308 | 0.2075 | 16 | 3.9 |
| 0.30 | 2 | 3 | 0.4128 | 0.6696 | 0.6166 | 0.0975 | 34 | 3.8 |
| 0.30 | 2 | 4 | 0.4097 | 0.6710 | 0.6107 | 0.0987 | 28 | 3.9 |
| 0.30 | 4 | 0 | 0.3036 | 0.3147 | 0.9645 | 0.2082 | 38 | 5.1 |
| 0.30 | 4 | 1 | 0.3136 | 0.3119 | 1.0054 | 0.1970 | 39 | 5.0 |
| 0.30 | 4 | 2 | 0.2856 | 0.3123 | 0.9145 | 0.2319 | 39 | 5.1 |
| 0.30 | 4 | 3 | 0.3074 | 0.3130 | 0.9822 | 0.2058 | 39 | 5.0 |
| 0.30 | 4 | 4 | 0.2508 | 0.3130 | 0.8014 | 0.2475 | 39 | 5.0 |

## Calls

| δ | call | L=2 R | n12 median Λ | detail |
|---:|---|---:|---:|---|
| 0.15 | PASS | 0.6130 | 0.7586 | L=2 R_Fourier ≤ 0.85 and median Λ ≤ 0.80 |
| 0.30 | PASS | 0.6167 | 0.7610 | L=2 R_Fourier ≤ 0.85 and median Λ ≤ 0.80 |

## Reading

δ=0.15: n12 median Λ=0.7586 (uniform 0.7477, shift +0.0109). Per-cavity medians 0.7884, 0.7444, 0.7413, 0.7686. L=2 R_Fourier=0.6130 (uniform 0.6201), 5/5 seeds ≤ 0.85. L=4 R_Fourier=0.8249. 26.8% of n=12 cavity-instance pairs sit above 0.80, 0.6% at or above 0.90. Median Z-peak=0.0334. Call **PASS**.

δ=0.30: n12 median Λ=0.7610 (uniform 0.7477, shift +0.0133). Per-cavity medians 0.8122, 0.7408, 0.7271, 0.7815. L=2 R_Fourier=0.6167 (uniform 0.6201), 5/5 seeds ≤ 0.85. L=4 R_Fourier=0.9699 (above 0.85; L=4 is not the gate). 31.4% of n=12 cavity-instance pairs sit above 0.80, 2.1% at or above 0.90. Median Z-peak=0.0312. Call **PASS**.

The closest approach to the Λ bar is the n=6 pack at δ=0.30, median Λ=0.7956, still ≤ 0.80. The gate uses the frozen n=12 pool, the same pool as the uniform 0.7477.

Disorder at 15% and 30% leaves the L=2 SNAP lead near the uniform ratio 0.62, on every seed, and it does not push the n=12 median cavity Λ through 0.80. The Λ-safe point stays Λ-safe once the transverse field is site-dependent.

L=4 is reported and is not the gate. The uniform L=4 ratio was already 0.869. δ=0.15 n12 R=0.825 (3/5 seeds ≤ 0.85), n6 holdout R=0.859. δ=0.30 n12 R=0.970 (1/5 seeds ≤ 0.85, 1 seed above 1), n6 holdout R=0.992. That depth is softer than L=2. It does not cross the L=2 kill line.

SNAP shells left the linear X mixer (δ=0.15 L=2 shell RMS 0.0773; δ=0.30 L=2 shell RMS 0.1246). Zero would be φ_k = a + b k.

Per-seed JSON is `{d0p15,d0p30}_L{L}_{method}_seed{s}.json`. Ground-state arrays are `gs_{tag}_n{n}_{start}_{stop}.npz`. `lambda_u.json` is the shared pattern.

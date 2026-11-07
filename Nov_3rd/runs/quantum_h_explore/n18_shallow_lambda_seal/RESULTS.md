# n=18 shallow seal — SAFE λ=0.75, μ=−0.50, with cavity Λ

**Verdict: KILL.** n18 median Λ > 0.85.

The shallow ratios clear the n=18 bars. L=2 R_HVA=0.2958 (bar ≤ 0.40), L=4 R_HVA=0.4010 (soft bar ≤ 0.65), both depths improve or hold from n=12, and the median product gap is 0.063 (bar ≥ 0.05). The lock is killed by the ground-state cavity diagnostic: median Λ=0.943 (bar ≤ 0.80, kill above 0.85). The same Λ is 0.941 on the n=12 freeze, so the collapse is a property of this SAFE point.

PR #38 sealed Walsh SNAP against HVA at frozen n=12 (L=2 R_HVA=0.328, L=4 R_HVA=0.536) and left n=18 as a 16-instance probe. The gate here is n=18, B=32, five seeds, L∈{2, 4} only. L≥8 is not trained. PR #36 remains the record that HVA takes L=8 and L=16.

## Gate

Med ε is the median over seeds of each seed's median-over-instances excess energy ⟨H⟩−E₀. R_HVA = Medε_SNAP / Medε_HVA. Degradation on a step is R_HVA(n+3) − R_HVA(n); a negative step is an improvement. The product gap is (E_prod − E_0) / |E_0| for the best real XZ product state (the SAFE median-gap definition). Λ_c = ⟨(Σ_{i∈c} X_i / 3)^2⟩ from the X histogram of the true ground state. The reported Λ is the median over cavities and instances.

- **KILL** when n=18 L=2 R_HVA > 0.50, or when median Λ > 0.85. Either one kills the SAFE lock.
- **PASS** when L=2 R_HVA ≤ 0.40, L=4 R_HVA ≤ 0.65, every +3-qubit step of R_HVA from n=12 is ≤ 0.05, the n=18 median product gap is ≥ 0.05, and median Λ ≤ 0.80.
- **needs_edit** when nothing kills the lock and at least one PASS bar is missed. L=4 and a mild Λ or gap miss are this soft case.

## Hamiltonian

\[
H = \mathrm{diag}(\mathrm{unsat}) - 0.75 \sum_i X_i - 0.50 \sum_{c}\sum_{i<j\in c} X_i X_j
\]

SAFE point **λ=0.75, μ=−0.50**. Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. Readout is the Z histogram (unsat) plus the X histogram (⟨X⟩, ⟨XX⟩, and Λ). No Y. The mixer is BinH Hamming-shell SNAP.

## Protocol

- Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Checkpoint on n=9 holdout [200:250). Freeze n=12 first 128 and n=15 first 64, the same slices as PR #38.
- n=18: 32 satisfiable formulas from `explore_OCT29.lib.sat_gen.instances` at base 61000000000. There is no n=18 train pack. These instances are not in the loss and not in checkpoint selection. The first 16 seeds match the PR #38 probe.
- L-BFGS matches `Nov_3rd.settings` and PR #38: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n6 / half n9). New optimizer each round. Loss = mean ε/n. Each seed restores the round with the best n=9-holdout Med ε.
- Paper init β=β^X=0.01, β^XX=0, γ=γ_w=−0.01, φ_k=−β k. The three inits are the same state.
- n=21 was not loaded. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.

## Ground state: Λ and product gap

Λ→1 means every cavity sits in a single X-magnetization sector, so the 3-qubit cavity has collapsed to an effective spin. The expectation going in was about 0.6–0.7. A median above 0.85 flags trivialization and kills the lock.

| n | B | median Λ | min Λ | max Λ | fraction Λ>0.85 | median product gap | min gap | fraction gap≥0.05 | median Z-peak |
|--:|--:|--------:|------:|------:|----------------:|-------------------:|--------:|------------------:|-------------:|
| 12 | 128 | 0.941 | 0.843 | 0.972 | 0.996 | 0.066 | 0.044 | 0.969 | 0.0036 |
| 15 | 64 | 0.942 | 0.838 | 0.977 | 0.988 | 0.063 | 0.045 | 0.969 | 0.0008 |
| 18 | 32 | 0.943 | 0.841 | 0.972 | 0.995 | 0.063 | 0.050 | 0.969 | 0.0002 |

n=18 median Λ = **0.943**. n=18 median product gap = **0.063**. The Z-basis peak of the ground state stays small (median 0.0002), so the state is not a classical bitstring. The large Λ is X-polarization inside each cavity, which is a different kind of collapse.

## L=2

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.1369 | 0.0952 | 0.0465 | 0.3399 | 0.4887 |
| 9 | holdout (select) | 0.1905 | 0.1600 | 0.0589 | 0.3091 | 0.3680 |
| 12 | freeze | 0.2664 | 0.2608 | 0.0855 | 0.3210 | 0.3280 |
| 15 | freeze | 0.3330 | 0.3504 | 0.1046 | 0.3143 | 0.2986 |
| 18 | freeze (gate) | 0.3993 | 0.4519 | 0.1336 | 0.3347 | 0.2958 |

**L=2 n18 R_HVA = 0.2958** (SNAP 0.1336 / HVA 0.4519). Steps of R_HVA: n12→n15 -0.0293 (0.3280 → 0.2986), n15→n18 -0.0029. Worst step -0.0029.

Paired n18 instances with SNAP ε below HVA: 160/160. HVA mean |β^XX| = 0.285.

| seed | n18 SNAP | n18 HVA | n18 Fourier | R_HVA | paired SNAP<HVA | n12 R_HVA | best round | wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1329 | 0.4566 | 0.4003 | 0.2911 | 32/32 | 0.3235 | 36 | 5 |
| 1 | 0.1341 | 0.4519 | 0.3997 | 0.2967 | 32/32 | 0.3290 | 28 | 5 |
| 2 | 0.1336 | 0.4314 | 0.3992 | 0.3098 | 32/32 | 0.3348 | 32 | 5 |
| 3 | 0.1329 | 0.4547 | 0.3993 | 0.2922 | 32/32 | 0.3250 | 15 | 5 |
| 4 | 0.1340 | 0.4451 | 0.3992 | 0.3011 | 32/32 | 0.3304 | 39 | 5 |

Circuit-state median Λ (X histogram of the prepared state, median over seeds of each seed's pooled median) at n18: Fourier 0.954, HVA 0.949, SNAP 0.948. The gate uses the ground-state Λ, not these.

## L=4

| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |
|---|---|---:|---:|---:|---:|---:|
| 6 | holdout | 0.0452 | 0.0229 | 0.0232 | 0.5142 | 1.0148 |
| 9 | holdout (select) | 0.0586 | 0.0354 | 0.0285 | 0.4866 | 0.8071 |
| 12 | freeze | 0.0771 | 0.0736 | 0.0394 | 0.5112 | 0.5356 |
| 15 | freeze | 0.0983 | 0.1108 | 0.0450 | 0.4574 | 0.4058 |
| 18 | freeze (gate) | 0.1173 | 0.1505 | 0.0604 | 0.5146 | 0.4010 |

**L=4 n18 R_HVA = 0.4010** (SNAP 0.0604 / HVA 0.1505). Steps of R_HVA: n12→n15 -0.1298 (0.5356 → 0.4058), n15→n18 -0.0048. Worst step -0.0048.

Paired n18 instances with SNAP ε below HVA: 160/160. HVA mean |β^XX| = 0.317.

| seed | n18 SNAP | n18 HVA | n18 Fourier | R_HVA | paired SNAP<HVA | n12 R_HVA | best round | wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0604 | 0.1452 | 0.1173 | 0.4157 | 32/32 | 0.5492 | 38 | 8 |
| 1 | 0.0619 | 0.1557 | 0.1177 | 0.3973 | 32/32 | 0.5345 | 39 | 8 |
| 2 | 0.0563 | 0.1506 | 0.1163 | 0.3740 | 32/32 | 0.5024 | 39 | 8 |
| 3 | 0.0804 | 0.1505 | 0.1167 | 0.5341 | 32/32 | 0.7194 | 39 | 8 |
| 4 | 0.0527 | 0.1495 | 0.1179 | 0.3522 | 32/32 | 0.4695 | 39 | 8 |

Circuit-state median Λ (X histogram of the prepared state, median over seeds of each seed's pooled median) at n18: Fourier 0.945, HVA 0.946, SNAP 0.949. The gate uses the ground-state Λ, not these.

## Reading

The call is **KILL**. n18 median Λ > 0.85.

Frozen n=12 Med ε reproduces PR #38 at both depths (L=2 R_HVA=0.3280, L=4 R_HVA=0.5356). On the new n=18 set the ratio keeps moving the right way: L=2 goes 0.3280 → 0.2986 → 0.2958, L=4 goes 0.5356 → 0.4058 → 0.4010. The largest step is below 0.05. SNAP is below HVA on every paired n=18 instance at both depths. β^XX is away from zero, so the HVA column carries a real XX mixer.

The ratio bars would pass on their own. Λ is what kills the lock. Median Λ is 0.941 / 0.942 / 0.943 at n=12 / 15 / 18, and more than 98% of cavities sit above 0.85. The ground state is spread out in the Z basis (n=18 median peak 0.0002), and it is almost a product of X-polarized cavity spins. Prepared states from Fourier, HVA, and SNAP land on the same Λ, near 0.95. The Hamming shell is acting on cavities that have already collapsed to an effective spin.

The product gap stays on the SAFE side of 0.05 (n=12 0.066, n=15 0.063, n=18 0.063). That bar is the one that separated SAFE μ=−0.50 from EDGE μ=−0.90. It stays cleared here, while Λ flags the X-polarization.

Λ is computed from the joint X histogram of the Lanczos ground state, one number per cavity. It equals 1/3 + (2/9) Σ_{i<j∈c} ⟨X_i X_j⟩. A Z-basis product state scores 1/3. The fully X-aligned cavity scores 1. Values near 1 mean the intra-cavity Hilbert space has collapsed to an effective spin, which is the trivialization this diagnostic was added to catch.

Per-seed JSON: `L{L}_{method}_seed{s}.json` in this folder. Angles are the restored n9-holdout checkpoint. Ground-state Λ and the product gap are in `diag_n{n}_{tag}.npz`.


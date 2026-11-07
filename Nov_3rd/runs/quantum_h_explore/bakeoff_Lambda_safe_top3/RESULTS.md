# Matched bakeoff — three Λ≤0.80 candidates

**Verdict: PASS. NEW_SAFE paper-lock is C, λ=0.75, μ=0.00 (baseline fourier).**

Round 12 trains the three cells that cleared Λ≤0.80 on the n=12 grid (PR #44 / PR #45) under one protocol. Walsh Hamming-shell SNAP is compared with the fair baseline: Fourier QAOA when μ=0, HVA when μ≠0. The killed SAFE point (λ=0.75, μ=−0.50, n=12 Λ≈0.941) is not retrained.

## Hamiltonian

\[
H = \mathrm{diag}(\mathrm{unsat}) - \lambda \sum_i X_i + \mu \sum_c \sum_{i<j\in c} X_i X_j
\]

Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. Λ is the median of ⟨(Σ_{i∈c} X_i / 3)^2⟩ over cavities and instances of the true ground state. The product gap is (E_prod − E_0) / |E_0| for the best real XZ product state found by the SAFE start set. Z-peak is the median over instances of max_x |⟨x|ψ_0⟩|^2. R = Medε_SNAP / Medε_baseline at n=12, L=2. Medε is the median over seeds of each seed's median-over-instances excess ⟨H⟩−E_0.

## Gate

- **Λ pass** when median Λ ≤ 0.80 at n=12 (B=128) and at n=18 (B=32).
- **PASS** a cell when it also has product gap ≥ 0.03 and Z-peak < 0.5 at both sizes, L=2 R ≤ 0.85, and all 5 seeds have per-seed R ≤ 0.85.
- **soft** when those quantumness bars hold and L=2 R ≤ 0.98, but the 5/5 bar does not.
- **FAIL** a cell when a quantumness bar misses, or the L=2 lead is above the soft band.
- **Family PASS** when at least one cell is PASS. Otherwise **needs_edit**.
- **NEW_SAFE** is the first PASS cell in the rank. Rank is Λ-pass first, then lowest L=2 R, then lowest n=12 Λ, then lowest n=18 Λ. A soft cell is not locked.
- L=4 is the same budget at one extra depth. It is reported and is not the lock.

## Protocol

Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Checkpoint on the n=9 holdout [200:250). Freeze n=12, the first 128 instances. The n=18 probe is the first 32 satisfiable formulas from the same generator and base (61000000000), seed = base + 10_000_000·n + k. It is stored in this run directory. `train_packs/` was not rewritten.

Seeds {0,1,2,3,4}. L∈{2,4}. One batch stream per seed: NumPy Generator(seed) and torch.manual_seed(seed), shared by every cell, both methods, and both depths. L-BFGS matches `Nov_3rd.settings`: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n=6 / half n=9). New optimizer each round. Loss = mean ε/n. Each seed restores the round with the best n=9-holdout Med ε. Paper init β=β^X=0.01, β^XX=0, γ=γ_w=−0.01, φ_k=−β k.

Ground states: dense eigh at n=6 and n=9, ARPACK at n=12 and n=18. n=21 was not loaded. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.

## Rank

| rank | id | λ | μ | baseline | Λ12 | Λ18 | gap12 | gap18 | Z-peak12 | Z-peak18 | L=2 R | 5/5 | call |
|--:|:--|---:|---:|:--|---:|---:|---:|---:|---:|---:|---:|:--|---|
| 1 | C | 0.75 | 0.00 | fourier | 0.748 | 0.749 | 0.127 | 0.134 | 0.0379 | 0.0064 | 0.620 | 5/5 | PASS |
| 2 | B | 0.50 | -0.15 | hva | 0.741 | 0.742 | 0.188 | 0.185 | 0.0497 | 0.0096 | 0.635 | 5/5 | PASS |
| 3 | A | 0.50 | 0.00 | fourier | 0.559 | 0.548 | 0.110 | 0.112 | 0.1569 | 0.0650 | 0.750 | 5/5 | PASS |

## Ground state

| id | n | B | median Λ | min Λ | max Λ | frac Λ>0.80 | frac Λ>0.85 | median gap | min gap | median Z-peak | frac peak≥0.5 |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 12 | 128 | 0.7477 | 0.5498 | 0.9117 | 0.230 | 0.061 | 0.1265 | 0.0450 | 0.0379 | 0.000 |
| C | 18 | 32 | 0.7487 | 0.5687 | 0.8748 | 0.203 | 0.062 | 0.1336 | 0.0786 | 0.0064 | 0.000 |
| B | 12 | 128 | 0.7410 | 0.5336 | 0.9129 | 0.215 | 0.049 | 0.1878 | 0.0612 | 0.0497 | 0.000 |
| B | 18 | 32 | 0.7417 | 0.5460 | 0.8748 | 0.198 | 0.057 | 0.1850 | 0.1129 | 0.0096 | 0.000 |
| A | 12 | 128 | 0.5585 | 0.4037 | 0.8593 | 0.010 | 0.002 | 0.1102 | 0.0413 | 0.1569 | 0.008 |
| A | 18 | 32 | 0.5477 | 0.3936 | 0.7656 | 0.000 | 0.000 | 0.1115 | 0.0460 | 0.0650 | 0.000 |

n=12 Λ on this pack, against the nomination value:

| id | Λ12 now | Λ12 prior | gap12 now | gap12 prior | Z-peak12 now | Z-peak12 prior |
|:--|---:|---:|---:|---:|---:|---:|
| C | 0.7477 | 0.7477 | 0.1265 | 0.1265 | 0.0379 | 0.0379 |
| B | 0.7410 | 0.7410 | 0.1878 | 0.1880 | 0.0497 | 0.0497 |
| A | 0.5585 | 0.5585 | 0.1102 | 0.1102 | 0.1569 | 0.1569 |

## Ratios

Holdout at n=6 and n=9 is the checkpoint distribution. The decision row is n=12, L=2.

| id | L | n | Medε baseline | Medε SNAP | R | paired SNAP<baseline | seeds ≤0.85 |
|:--|---:|---:|---:|---:|---:|---:|---:|
| C | 2 | 6 | 0.3580 | 0.2196 | 0.6135 | 250/250 |  |
| C | 2 | 9 | 0.4722 | 0.2978 | 0.6307 | 250/250 |  |
| C | 2 | 12 | 0.6876 | 0.4264 | 0.6201 | 640/640 | 5/5 |
| C | 4 | 6 | 0.1452 | 0.1388 | 0.9555 | 180/250 |  |
| C | 4 | 9 | 0.2254 | 0.1977 | 0.8773 | 226/250 |  |
| C | 4 | 12 | 0.3172 | 0.2755 | 0.8687 | 581/640 | 2/5 |
| B | 2 | 6 | 0.3329 | 0.2451 | 0.7362 | 240/250 |  |
| B | 2 | 9 | 0.4834 | 0.3347 | 0.6924 | 250/250 |  |
| B | 2 | 12 | 0.7246 | 0.4602 | 0.6350 | 640/640 | 5/5 |
| B | 4 | 6 | 0.1366 | 0.1543 | 1.1302 | 52/250 |  |
| B | 4 | 9 | 0.2268 | 0.2236 | 0.9856 | 164/250 |  |
| B | 4 | 12 | 0.3416 | 0.2964 | 0.8677 | 619/640 | 2/5 |
| A | 2 | 6 | 0.6044 | 0.4446 | 0.7357 | 243/250 |  |
| A | 2 | 9 | 0.8907 | 0.6890 | 0.7736 | 245/250 |  |
| A | 2 | 12 | 1.1682 | 0.8765 | 0.7503 | 640/640 | 5/5 |
| A | 4 | 6 | 0.3197 | 0.3061 | 0.9575 | 177/250 |  |
| A | 4 | 9 | 0.5015 | 0.4827 | 0.9625 | 206/250 |  |
| A | 4 | 12 | 0.6466 | 0.5958 | 0.9214 | 552/640 | 0/5 |

### Seeds at n=12, L=2

| id | seed | SNAP | baseline | R | best round | wall s |
|:--|---:|---:|---:|---:|---:|---:|
| C | 0 | 0.4264 | 0.6876 | 0.6202 | 35 | 4.0 |
| C | 1 | 0.4243 | 0.6878 | 0.6169 | 32 | 4.2 |
| C | 2 | 0.4264 | 0.6876 | 0.6201 | 31 | 4.2 |
| C | 3 | 0.4261 | 0.6876 | 0.6196 | 34 | 4.2 |
| C | 4 | 0.4274 | 0.6878 | 0.6215 | 34 | 4.5 |
| B | 0 | 0.4602 | 0.7260 | 0.6338 | 36 | 4.9 |
| B | 1 | 0.4594 | 0.7218 | 0.6365 | 39 | 5.1 |
| B | 2 | 0.4638 | 0.7246 | 0.6401 | 32 | 5.0 |
| B | 3 | 0.4588 | 0.7238 | 0.6339 | 38 | 4.8 |
| B | 4 | 0.4620 | 0.7255 | 0.6368 | 39 | 5.0 |
| A | 0 | 0.8764 | 1.1689 | 0.7498 | 38 | 6.2 |
| A | 1 | 0.8752 | 1.1682 | 0.7492 | 39 | 6.4 |
| A | 2 | 0.8788 | 1.1687 | 0.7519 | 34 | 6.2 |
| A | 3 | 0.8776 | 1.1679 | 0.7514 | 35 | 6.3 |
| A | 4 | 0.8765 | 1.1682 | 0.7503 | 39 | 4.6 |

### Seeds at n=12, L=4

| id | seed | SNAP | baseline | R | best round | wall s |
|:--|---:|---:|---:|---:|---:|---:|
| C | 0 | 0.2755 | 0.3184 | 0.8653 | 38 | 5.4 |
| C | 1 | 0.3078 | 0.3169 | 0.9714 | 39 | 5.4 |
| C | 2 | 0.2534 | 0.3172 | 0.7989 | 38 | 5.5 |
| C | 3 | 0.2777 | 0.3163 | 0.8778 | 39 | 5.6 |
| C | 4 | 0.2241 | 0.3181 | 0.7044 | 39 | 5.5 |
| B | 0 | 0.2964 | 0.3416 | 0.8677 | 38 | 6.1 |
| B | 1 | 0.3147 | 0.3401 | 0.9252 | 39 | 5.8 |
| B | 2 | 0.2783 | 0.3401 | 0.8184 | 38 | 5.9 |
| B | 3 | 0.2993 | 0.3420 | 0.8754 | 39 | 5.9 |
| B | 4 | 0.2770 | 0.3424 | 0.8088 | 39 | 5.8 |
| A | 0 | 0.5958 | 0.6473 | 0.9204 | 39 | 7.1 |
| A | 1 | 0.6096 | 0.6459 | 0.9439 | 39 | 6.9 |
| A | 2 | 0.5747 | 0.6441 | 0.8922 | 39 | 6.9 |
| A | 3 | 0.6179 | 0.6499 | 0.9507 | 39 | 6.5 |
| A | 4 | 0.5646 | 0.6466 | 0.8732 | 39 | 6.5 |

## Recommendation

Paper-lock **C** as NEW_SAFE: λ=0.75, μ=0.00, baseline fourier.

n=12 Λ=0.748, n=18 Λ=0.749, n=12 gap=0.127, n=18 gap=0.134, n=12 Z-peak=0.0379, n=18 Z-peak=0.0064, L=2 R=0.620 on 5/5 seeds.

It is the Λ-passing cell with the lowest L=2 R among cells that also clear the 5/5 bar. It is rank 1.

C clears the median bar with 23.0% of n=12 cavity-instance pairs and 20.3% at n=18 still above 0.80. A has the lowest cavity Λ (n12 0.559, n18 0.548) and a higher L=2 R (0.750). Among cells that clear Λ≤0.80, the rank uses the lowest R. L=4 stays in the soft band (C 0.869 on 2/5, B 0.868 on 2/5, A 0.921 on 0/5). The paper-lock uses L=2. On the n=6 holdout at L=4, SNAP is above the baseline for B (R=1.130).

## Reading

- **C** (λ=0.75, μ=0.00, fourier): PASS. Λ12=0.748, Λ18=0.749, gap12=0.127, gap18=0.134, Z-peak12=0.0379, Z-peak18=0.0064, L=2 R=0.620 (5/5), L=4 R=0.869 (2/5).
- **B** (λ=0.50, μ=-0.15, hva): PASS. Λ12=0.741, Λ18=0.742, gap12=0.188, gap18=0.185, Z-peak12=0.0497, Z-peak18=0.0096, L=2 R=0.635 (5/5), L=4 R=0.868 (2/5). HVA mean |β^XX|=0.195.
- **A** (λ=0.50, μ=0.00, fourier): PASS. Λ12=0.559, Λ18=0.548, gap12=0.110, gap18=0.112, Z-peak12=0.1569, Z-peak18=0.0650, L=2 R=0.750 (5/5), L=4 R=0.921 (0/5).

n=12 Λ, the product gap, and the Z-peak reproduce the nomination (A and C from the μ=0 seal, B from the three-seed grid) to 0.001.

The family call is **PASS**.

Per-seed JSON is `runs/L{L}_{method}_s{seed}_{id}.json`. Ground-state arrays are `cache/diag_{id}_n{n}.npz`. The n=18 formulas are `probe_n18.npz`.

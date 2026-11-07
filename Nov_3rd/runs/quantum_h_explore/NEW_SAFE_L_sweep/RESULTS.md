# NEW_SAFE depth sweep — Walsh SNAP vs Fourier

**Verdict: PASS.** L=2 frozen n=12 R_Fourier=0.620, median Λ(GS)=0.748.

Point **NEW_SAFE**: λ=0.75, μ=0. Baseline is Fourier QAOA (no XX mixer; μ=0). Old SAFE (λ=0.75, μ=−0.50) is not retrained. That point was cavity-trivial (Λ≈0.94) and HVA took the deep depths.

## Hamiltonian

\[
H = \mathrm{diag}(\mathrm{unsat}) - 0.75 \sum_i X_i
\]

μ=0, so there is no cavity XX term. Cavity c owns qubits [3c, 3c+3). ε = ⟨H⟩ − E₀ from the Z histogram (unsat) and the X histogram (⟨X⟩). Medε is the median over seeds of each seed's median-over-instances ε. R_Fourier = Medε_SNAP / Medε_Fourier. Λ is the median of ⟨(Σ_{i∈c} X_i / 3)²⟩ over cavities and instances of the true ground state. It does not depend on L.

## Gate

- Train n∈{6,9}, indices [0:200). Checkpoint on the n=9 holdout [200:250).
- Freeze n=12, the first 128 pack instances. n=21 was not loaded.
- **PASS** when frozen-n12 L=2 R ≤ 0.85 and n=12 median Λ ≤ 0.80.
- **win** at a depth when R ≤ 0.85. **soft** when 0.85 < R < 0.98. **tie** when 0.98 ≤ R ≤ 1.02. **loss** when R > 1.02.
- A depth with R ≥ 0.98 is not a SNAP win.

## Protocol

Packs: `beat_qaoa_12h/train_packs`. Seeds depend on L: L∈{1,2,4} use {0,1,2,3,4}; L=8 uses {0,1,2}; L=16 uses {0,1}. One batch stream per seed (NumPy Generator(seed) and torch.manual_seed(seed)), shared by both methods and every depth. L-BFGS matches `Nov_3rd.settings`: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64. New optimizer each round. Loss = mean ε/n. Each seed restores the round with the best n=9-holdout Medε. Paper init β=0.01, γ=γ_w=−0.01, φ_k=−β k. With an even seed count (L=16), Medε is the mean of the two seed medians. Ground states: dense eigh at n=6 and n=9, ARPACK at n=12. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.

## Frozen n=12

| L | seeds | Medε SNAP | Medε Fourier | R_Fourier | median Λ (GS) | paired SNAP<Fourier | seeds R≤0.85 | call |
|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 1 | 5 | 0.9762 | 1.1177 | 0.8734 | 0.7477 | 639/640 | 0/5 | soft |
| 2 | 5 | 0.4264 | 0.6876 | 0.6201 | 0.7477 | 640/640 | 5/5 | win |
| 4 | 5 | 0.2755 | 0.3172 | 0.8687 | 0.7477 | 581/640 | 2/5 | soft |
| 8 | 3 | 0.2255 | 0.1173 | 1.9232 | 0.7477 | 0/384 | 0/3 | loss |
| 16 | 2 | 0.1726 | 0.0797 | 2.1641 | 0.7477 | 0/256 | 0/2 | loss |

Holdout Medε (checkpoint distribution, B=50). Λ there is the ground state on the full pack prefix that was diagonalized (n=6 and n=9, B=250), not a function of L.

| L | n | Medε SNAP | Medε Fourier | R_Fourier | median Λ (GS) | paired SNAP<Fourier | call |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 1 | 6 | 0.4700 | 0.5508 | 0.8534 | 0.7612 | 249/250 | soft |
| 1 | 9 | 0.6801 | 0.7791 | 0.8729 | 0.7555 | 250/250 | soft |
| 2 | 6 | 0.2196 | 0.3580 | 0.6135 | 0.7612 | 250/250 | win |
| 2 | 9 | 0.2978 | 0.4722 | 0.6307 | 0.7555 | 250/250 | win |
| 4 | 6 | 0.1388 | 0.1452 | 0.9555 | 0.7612 | 180/250 | soft |
| 4 | 9 | 0.1977 | 0.2254 | 0.8773 | 0.7555 | 226/250 | soft |
| 8 | 6 | 0.1141 | 0.0528 | 2.1623 | 0.7612 | 3/150 | loss |
| 8 | 9 | 0.1733 | 0.0838 | 2.0694 | 0.7555 | 0/150 | loss |
| 16 | 6 | 0.0866 | 0.0371 | 2.3306 | 0.7612 | 0/100 | loss |
| 16 | 9 | 0.1292 | 0.0571 | 2.2620 | 0.7555 | 0/100 | loss |

## Ground state

Λ and the Z-peak do not change with L. The n=12 row is the gate.

| n | B | median Λ | min Λ | max Λ | frac Λ>0.80 | median Z-peak | frac peak≥0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 250 | 0.7612 | 0.5741 | 0.9417 | 0.262 | 0.2022 | 0.000 |
| 9 | 250 | 0.7555 | 0.5223 | 0.9343 | 0.247 | 0.0916 | 0.000 |
| 12 | 128 | 0.7477 | 0.5498 | 0.9117 | 0.230 | 0.0379 | 0.000 |

## Seeds at frozen n=12

### L=1

| seed | Medε SNAP | Medε Fourier | R | best round | wall s |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9827 | 1.1181 | 0.8790 | 19 | 3.0 |
| 1 | 0.9762 | 1.1176 | 0.8735 | 10 | 3.0 |
| 2 | 0.9818 | 1.1177 | 0.8784 | 9 | 3.0 |
| 3 | 0.9756 | 1.1184 | 0.8723 | 8 | 3.1 |
| 4 | 0.9751 | 1.1169 | 0.8731 | 14 | 3.1 |

### L=2

| seed | Medε SNAP | Medε Fourier | R | best round | wall s |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.4264 | 0.6876 | 0.6202 | 35 | 3.6 |
| 1 | 0.4243 | 0.6878 | 0.6169 | 32 | 3.7 |
| 2 | 0.4264 | 0.6876 | 0.6201 | 31 | 3.6 |
| 3 | 0.4261 | 0.6876 | 0.6196 | 34 | 3.6 |
| 4 | 0.4274 | 0.6878 | 0.6215 | 34 | 3.6 |

### L=4

| seed | Medε SNAP | Medε Fourier | R | best round | wall s |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2755 | 0.3184 | 0.8653 | 38 | 5.0 |
| 1 | 0.3078 | 0.3169 | 0.9714 | 39 | 5.1 |
| 2 | 0.2534 | 0.3172 | 0.7989 | 38 | 5.1 |
| 3 | 0.2777 | 0.3163 | 0.8778 | 39 | 5.5 |
| 4 | 0.2241 | 0.3181 | 0.7044 | 39 | 5.4 |

### L=8

| seed | Medε SNAP | Medε Fourier | R | best round | wall s |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2178 | 0.1221 | 1.7830 | 38 | 7.6 |
| 1 | 0.2360 | 0.1173 | 2.0130 | 39 | 8.4 |
| 2 | 0.2255 | 0.1133 | 1.9902 | 38 | 8.6 |

### L=16

| seed | Medε SNAP | Medε Fourier | R | best round | wall s |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.1755 | 0.0792 | 2.2174 | 38 | 12.7 |
| 1 | 0.1697 | 0.0803 | 2.1116 | 39 | 12.6 |

## Paper note

SNAP wins at L=2. L=1 and L=4 stay soft. L=8 and L=16 lose to Fourier, so they are not a deep win.

Median Λ(GS)=0.748 at frozen n=12 (n=6 0.7612, n=9 0.7555). Λ is a property of the ground state, so the column is the same at every L. The gate is Λ≤0.80. At n=12 the median Z-peak is 0.0379 and 23% of cavity-instance pairs still sit above 0.80.

L=2 wins: Medε SNAP 0.4264 vs Fourier 0.6876, R=0.620, 5/5 seeds ≤0.85, SNAP lower on 640/640 paired instances. This is the shallow confirmation (the prior lock was R≈0.62).

The L=2 win is also on the holdout (n=6 R=0.613, n=9 R=0.631).

L=1 is soft, not a win: R=0.873, 0/5 seeds ≤0.85. SNAP is still lower on 639/640 instances, but the medians are too close for the 0.85 bar. One shell layer is not the figure.

L=4 is soft: R=0.869, 2/5 seeds ≤0.85, paired 581/640. The n=6 holdout at L=4 is R=0.956, inside the soft band.

L≥8 loses. L=8 R=1.923 (loss, SNAP lower on 0/384, 3 seeds); L=16 R=2.164 (loss, SNAP lower on 0/256, 2 seeds). Fourier at L=8 (Medε 0.1173) is already below SNAP at L=16 (Medε 0.1726). Both errors fall with depth; after L=2, Fourier falls faster. R≥0.98 is not a deep win. L=16 is a two-seed check, not a five-seed seal.

n=21 was not run.

Per-seed JSON is `runs/L{L}_{method}_s{seed}.json`. Ground-state arrays are `cache/diag_n{n}.npz`.

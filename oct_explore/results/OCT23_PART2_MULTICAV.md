# Oct 23 Part 2: Multi-Cavity 4-SAT Scaling (`C = 2..5` Cavities, `n = 8, 12, 16, 20`, Physical SNAP+D + Global Cost + Pure GNN)

*Last updated: 2026-10-23 22:50:56 (UTC+8). Every cavity holds `K = 4` bits in `D = 16` computational Fock levels (`0..15`, Gray code); all SNAP gates restricted to `0..15` (`0` phase on leaked levels `f >= 16`); displacements `D(alpha)` computed in `N_Fock = 40` and projected onto `0..15` at each layer (strictly no refocusing from leaked levels). Zero classical pre-solving (`relabel = none`, `fv[:,:,3:7] = 0`).*

## 1. Classical WalkSATlm & Ideal-Mixer Reference Baselines on the 100 Unseen Test Formulas (`TEB = 52e9`)

| Method | `n = 8` (`C=2`) Med / B90 | `n = 12` (`C=3`) Med / B90 | `n = 16` (`C=4`) Med / B90 | `n = 20` (`C=5`) Med / B90 | Growth `b^n` (Med / B90) |
|---|---|---|---|---|---|
| **WalkSATlm (lookups)** (`p=0.30`) | **14.0 / 29.4** | **32.5 / 76.1** | **71.0 / 147.0** | **110.0 / 290.0** | **1.191^n / 1.207^n** |
| **WalkSATlm (tries)** (`p=0.30`) | **2.01 / 3.74** | **2.92 / 6.28** | **4.55 / 9.09** | **5.63 / 14.20** | **1.092^n / 1.116^n** |
| *(ref)* Ideal Mixer QAOA L4 (`OCT23_DATA.md`) | 5.78 / 10.3 | 12.8 / 33.1 | 34.1 / 84.9 | 77.6 / 198.0 | 1.245^n / 1.278^n |
| *(ref)* Ideal Mixer Global Cost L4 (`OCT23_DATA.md`) | 5.27 / 9.12 | 11.5 / 26.5 | 31.7 / 69.7 | 66.1 / 172.0 | 1.240^n / 1.277^n |

## 2. Physical SNAP+D Multi-Cavity Scaling Runs (`n = 8, 12` Train -> `n = 8, 12, 16, 20` Unseen Test)

| Tag | Mode (`Arch`) | `L` (`k_blk`) | Cost | Val12 Med / B90 | `n = 8` Med / B90 | `n = 12` Med / B90 | `n = 16` Med / B90 | `n = 20` Med / B90 | Growth `b^n` (Med / B90) |
|---|---|---|---|---|---|---|---|---|---|

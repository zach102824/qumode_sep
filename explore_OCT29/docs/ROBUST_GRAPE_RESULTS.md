# Robust GRAPE probes — verdict

_Generated 2026-10-31 00:00:35  (Asia/Shanghai)_

Doc: `docs/ROBUST_GRAPE.md`. Key column: **=exp(Hc)?**

## Verdict table (L=16, train{9,12}, scale1000 B=1000)

| arm | =exp(Hc)? | instance in cost? | n9 Med/B90 | n12 | n15 | vs ArmC n15 |
|-----|-----------|-------------------|------------|-----|-----|-------------|
| Arm C densify | YES | full E | 1.441/3.314 | 1.991/6.252 | 2.921/10.959 | ×1.0000 |
| projector (=Hc) | YES | clause→C | 1.406/3.288 | 1.864/6.106 | 2.754/10.244 | ×1.0606 |
| parity (FAILED) | NO | XOR clauses | 309.767/2158.960 | 1912.569/11831.688 | 13150.490/87785.854 | ×0.0002 |
| stencil_x (FAILED) | NO | none | 256.000/512.000 | 2048.001/4096.005 | 16383.996/32768.035 | ×0.0002 |
| mixer_only | NO | none | 256.001/512.001 | 2048.005/4096.012 | 16384.037/32768.109 | ×0.0002 |
| random_diag | NO | none (fake E) | 314.054/2536.888 | 2145.215/14285.184 | 13972.642/88453.254 | ×0.0002 |
| shuffle_E | NO | scrambled E | 1796.931/13156.371 | 5864.238/57137.478 | 24929.088/233820.573 | ×0.0001 |
| universal_cover | NO (global phase) | none | 256.000/512.001 | 2048.002/4096.005 | 16384.018/32768.074 | ×0.0002 |
| trunc_Hc | NO (partial) | m/2 clauses | 20.796/64.953 | 63.160/214.865 | 192.176/912.036 | ×0.0152 |

## Interpretation

- **Projector product = exp(-iγ Hc)** for TEB 4-SAT. Same Med as Arm C is expected; it only avoids *densifying* E, not using Hc.
- **Ablation trio** (mixer_only / random_diag / shuffle_E): Med ≫ projector (n15 ~1e4–2e4 vs ~2.8). GRAPE+shared angles do **not** invent SAT structure without cost.
- **universal_cover** (all 26 locals / cavity / layer): multiplicity ≡ 7 on every Fock → **global phase only** ≡ mixer_only. Confirmed Med = 2^{n-1}.
- **trunc_Hc** (first m/2 clauses): better than blind (n15 Med≈192) but still ~70× worse than Arm C/projector. Partial Hc ≠ full Hc.
- **Hardware story that works:** frozen clause-projector library (= Uc) + shared γ/φ. That **renames** Hc for GRAPE compile; it is not Hc-free physics.
- Parked: surrogate / feature-schedule (Opus greenlight: after trio).

## Artifacts

- Params: `runs/robust_grape/*_params.pt`
- JSON: `runs/robust_grape/results.json`
- Trio backup: `results_trio.json`


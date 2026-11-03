# 3p gauss SNAP L=1 vs QAOA L=1 — scale1000 B=1000

_Generated 2026-11-03 10:56:40 Asia/Shanghai_

**Ansatz (ours):** Nov1st-final, `snap_mode=gauss` (3 params/L: α+β·pc+δ·pc²), L=1, per-layer
**Train:** fresh L=1 (not reshape of L=32). tag=`gauss_L1_cvar08_gauss_trunc_s10`
**Protocol:** train n∈{6,9} max_train=200; loss=cvar q=0.8; Adam+LBFGS; warm=gauss_trunc
**QAOA L=1:** ideal qubit QAOA from `qaoa_L1_RESULTS.md` (Med n9/12/15/18 = 29.279/90.033/289.892/920.513)

## Ladder (1/p) — 3p gauss L=1

| n | Med | B90 | p(gs)≈1/Med |
|---|-----|-----|-------------|
| 9 | 55.883 | 117.210 | 0.017895 |
| 12 | 249.207 | 544.954 | 0.004013 |
| 15 | 1092.997 | 2565.851 | 0.000915 |
| 18 | 4342.352 | 12066.282 | 0.000230 |

## Side-by-side vs QAOA L=1

| n | gauss Med | gauss B90 | gauss p(gs) | QAOA Med | QAOA B90 | QAOA p(gs) | gauss÷QAOA Med | gauss÷QAOA B90 |
|---|-----------|-----------|-------------|----------|----------|------------|----------------|----------------|
| 9 | 55.883 | 117.210 | 0.017895 | 29.279 | 63.429 | 0.034154 | **1.909×** | 1.848× |
| 12 | 249.207 | 544.954 | 0.004013 | 90.033 | 216.627 | 0.011107 | **2.768×** | 2.516× |
| 15 | 1092.997 | 2565.851 | 0.000915 | 289.892 | 750.514 | 0.003450 | **3.770×** | 3.419× |
| 18 | 4342.352 | 12066.282 | 0.000230 | 920.513 | 2585.026 | 0.001086 | **4.717×** | 4.668× |

## Med-only (for chat)

| n | QAOA L=1 | 3p gauss L=1 | gauss÷QAOA |
|---|----------|--------------|------------|
| 9 | 29.3 (p≈0.0342) | 55.9 (p≈0.0179) | **1.91×** |
| 12 | 90.0 (p≈0.0111) | 249.2 (p≈0.0040) | **2.77×** |
| 15 | 289.9 (p≈0.0034) | 1093.0 (p≈0.0009) | **3.77×** |
| 18 | 920.5 (p≈0.0011) | 4342.4 (p≈0.0002) | **4.72×** |

## b fits (Med ~ a·b^n on n=9..18)

- **3p gauss L=1** b_Med≈**1.6235**
- **QAOA L=1** b_Med≈**1.4678**

## Notes

- This is the SNAP-easy BEAT ansatz family (`gp_cvar08_from_gauss_s0` at L=32) retrained fresh at L=1.
- Arm C / Nov progressive shell are intentionally omitted; comparison is vs QAOA L=1 only.
- Champ path: `/workspace/qumode_sep/Nov/runs/snap_easy/gauss_L1_vs_qaoa/ckpts/params_gauss_L1_cvar08_gauss_trunc_s10_L1_seed10.pt`


# 3p gauss SNAP L=1 vs QAOA L=1

Canonical writeup: `/workspace/qumode_sep/Nov/runs/snap_easy/gauss_L1_vs_qaoa/RESULTS.md`

Champ: `gauss_L1_cvar08_gauss_trunc_s10` (fresh L=1 train, cvar q=0.8, warm=gauss_trunc layer-0 from L32 gauss champ).

| n | QAOA Med | 3p gauss Med | gauss÷QAOA | QAOA p(gs) | gauss p(gs) |
|---|----------|--------------|------------|------------|-------------|
| 9 | 29.279 | 55.883 | **1.909×** | 0.034154 | 0.017895 |
| 12 | 90.033 | 249.207 | **2.768×** | 0.011107 | 0.004013 |
| 15 | 289.892 | 1092.997 | **3.770×** | 0.003450 | 0.000915 |
| 18 | 920.513 | 4342.352 | **4.717×** | 0.001086 | 0.000230 |

b_Med: gauss L=1 ≈1.6235 · QAOA L=1 ≈1.4678

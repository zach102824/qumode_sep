# SNAP easiness — RESULTS

_Updated 2026-11-02 12:57 Asia/Shanghai_

Motif locked: Nov1st-final. Train n∈{6,9} max_train=200. Eval scale1000 B=1000.

Sealed: n9 1.092/1.776, n12 1.197/2.843, n15 1.418/5.505, n18 1.752/11.353.

## ★ BEAT_SEALED — gauss CVaR polish

Tag: `gp_cvar08_from_gauss_s0` (gauss perL, 3 params/L, CVaR q=0.8 Adam600+LBFGS warm from near-beat s0).

Checkpoint: `gauss_polish/gauss/params_gp_cvar08_from_gauss_s0_L32_seed0.pt`

JSON: `BEAT_SEALED.json`

| n | Med / B90 | ×Med / ×B90 |
|---|-----------|-------------|
| 9 | **1.080 / 1.756** | **0.989× / 0.989×** |
| 12 | **1.167 / 2.833** | **0.975× / 0.997×** |
| 15 | **1.347 / 5.017** | **0.950× / 0.911×** |
| 18 | **1.623 / 10.523** | **0.926× / 0.927×** |

**BEAT_SEALED (restricted SNAP dual-beat all n): YES**

## Full B=1000 finalists

| tag | params | n9 | n12 | n15 | n18 | beat? |
|-----|--------|----|-----|-----|-----|-------|
| `easy_snap_few_mM2_perL_L32_s0_full` | 2 | 1.138/1.988 (1.042×/1.119×) | 1.381/4.003 (1.154×/1.408×) | 1.881/7.824 (1.326×/1.421×) | 2.746/20.737 (1.567×/1.827×) | NO |
| `easy_snap_gauss_perL_L32_s0_full` | 3 | 1.076/1.791 (0.986×/1.008×) | 1.167/2.864 (0.975×/1.007×) | 1.340/4.934 (0.945×/0.896×) | 1.620/10.811 (0.925×/0.952×) | NO |
| `easy_snap_poly2_n_perL_L32_s1_full` | 3 | 1.185/2.634 (1.085×/1.483×) | 1.407/6.173 (1.175×/2.171×) | 1.820/10.466 (1.284×/1.901×) | 2.506/32.942 (1.430×/2.902×) | NO |
| `gp_cvar08_from_gauss_s0_full` | 3 | 1.080/1.756 (0.989×/0.989×) | 1.167/2.833 (0.975×/0.997×) | 1.347/5.017 (0.950×/0.911×) | 1.623/10.523 (0.926×/0.927×) | YES |

## Screen B=200 (queue / non-full)

| tag | params | n9 | n12 |
|-----|--------|----|-----|
| `easy_snap_few_mM3_perL_L32_s0_screenB200` | 3 | 1.120/1.901 (1.026×/1.071×) | 1.431/2.971 (1.196×/1.045×) |
| `easy_snap_few_mM3_shared_L32_s0_screenB200` | 3 | 1.438/2.159 (1.317×/1.215×) | 1.904/3.389 (1.590×/1.192×) |
| `easy_snap_gauss_plus1_perL_L32_s1_screenB200` | 4 | 1.074/1.694 (0.984×/0.954×) | 1.190/2.347 (0.994×/0.825×) |
| `easy_snap_gauss_plus1_shared_L32_s0_screenB200` | 4 | 1.410/2.143 (1.291×/1.207×) | 1.822/3.318 (1.522×/1.167×) |
| `easy_snap_gauss_shared_L32_s0_screenB200` | 3 | 1.436/2.141 (1.315×/1.205×) | 1.855/3.346 (1.550×/1.177×) |
| `easy_snap_poly2_n_shared_L32_s1_screenB200` | 3 | 1.815/3.817 (1.662×/2.149×) | 2.527/6.727 (2.111×/2.366×) |
| `easy_snap_poly3_pc_perL_L32_s1_screenB200` | 4 | 1.086/1.666 (0.994×/0.938×) | 1.245/2.324 (1.041×/0.818×) |
| `easy_snap_shell_perL_L32_s0_screenB200` | 4 | 1.081/1.557 (0.990×/0.877×) | 1.245/2.256 (1.040×/0.794×) |

## Most-restricted survivors (n9 Med ≤ 1.10× sealed, any eval)

- **few_m M=2** (2 params/L): full n9 1.138/1.988 (1.042×/1.119×) — most restricted survivor; B90 short
- **gauss** (3): CVaR polish **BEATS** sealed dual all n
- **poly2_n** (3): full worse B90 (~1.48–2.9×)
- **gauss_plus1 / poly3_pc / shell** (4): screen strong; full B=1000 still queued

Stop file: `Nov/runs/STOP_SNAP_EASY`.


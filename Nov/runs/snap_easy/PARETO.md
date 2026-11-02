# SNAP-easy pareto — BEAT_SEALED

_Updated 2026-11-02 12:57 Asia/Shanghai_

## Clear pareto + sealed dual-beat: `gauss` CVaR polish (3 params/L)

Warm-start from `easy_snap_gauss_perL_L32_s0` (which was ~1.007–1.008× B90 short at n9/12),
then CVaR q=0.8 Adam 600 + LBFGS → `gp_cvar08_from_gauss_s0`.

Checkpoint: `gauss_polish/gauss/params_gp_cvar08_from_gauss_s0_L32_seed0.pt`

JSON: `BEAT_SEALED.json` / `gauss_polish/scale1000_gp_cvar08_from_gauss_s0_full.json`

Sealed: n9 1.092/1.776 · n12 1.197/2.843 · n15 1.418/5.505 · n18 1.752/11.353

| n | Med / B90 | ×Med / ×B90 | vs sealed |
|---|-----------|-------------|-----------|
| 9 | **1.080 / 1.756** | **0.989× / 0.989×** | dual beat |
| 12 | **1.167 / 2.833** | **0.975× / 0.997×** | dual beat |
| 15 | **1.347 / 5.017** | **0.950× / 0.911×** | dual beat |
| 18 | **1.623 / 10.523** | **0.926× / 0.927×** | dual beat |

**BEAT_SEALED (Med+B90 all n): YES**

Hardware: gauss = θ_k = α + β·pc(k) + δ·pc(k)² — 3 params/layer.

## Queue

Ladder 714841 still finishing full B=1000 on few_mM3, gauss_plus1, poly3_pc, shell (and later waves for kerr/more seeds).


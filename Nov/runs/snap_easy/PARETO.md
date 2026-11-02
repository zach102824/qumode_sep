# SNAP-easy pareto — full B=1000

_Updated 2026-11-02 12:17 Asia/Shanghai_

## Clear pareto update: `gauss` perL (3 params/L)

Most-restricted head that stays near sealed with **complete** scale1000 B=1000 on n∈{9,12,15,18}.

Checkpoint: `snap_gauss_perL/params_easy_snap_gauss_perL_L32_s0_L32_seed0.pt`
JSON: `scale1000_easy_snap_gauss_perL_L32_s0_full.json`

Sealed: n9 1.092/1.776 · n12 1.197/2.843 · n15 1.418/5.505 · n18 1.752/11.353

| n | Med / B90 | ×Med / ×B90 | vs sealed |
|---|-----------|-------------|-----------|
| 9 | **1.076 / 1.791** | **0.986× / 1.008×** | Med beat; B90 miss ~0.8% |
| 12 | **1.167 / 2.864** | **0.975× / 1.007×** | Med beat; B90 miss ~0.7% |
| 15 | **1.340 / 4.934** | **0.945× / 0.896×** | dual beat |
| 18 | **1.620 / 10.811** | **0.925× / 0.952×** | dual beat |

- **Med ≤ sealed at all four n**
- **B90 ≤ sealed at n15 & n18**; n9/n12 B90 ≈ 1.007–1.008× (hair short of dual-beat)
- **BEAT_SEALED (Med+B90 all n): NO**

Hardware note: `gauss` = θ_k = α + β·pc(k) + δ·pc(k)² — 3 params/layer (smoother than shell’s 4, far easier than free’s 8). Kerr (2 params) still weaker on B90 (partial n9/12 only).

## Queue

Full B=1000 still running for: poly2_n, few_m M=2/3, gauss_plus1, poly3_pc, shell.
Hybrid/wave watchdog continues nonstop (honor `Nov/runs/STOP_SNAP_EASY`).

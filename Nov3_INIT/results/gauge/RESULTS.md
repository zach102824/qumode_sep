# SK spin-flip gauge robustness N=15 L=1

Original shared-Uc Nov3 vs QAOA p=1. Frozen v4 center-4. Identity + 3 frozen random local spin-flip gauges. Statistical unit: 8 instances, gauges correlated. Exploratory; no adoption gate; no GR. Compiler stream closed. Hardware W_L unvalidated.

**GAUGE_EXPLORATORY.**

New energy queries 65536. Historical 2818048. Total 2883584. Failures 0.
Classical B* wall 63.1s. QAOA cross-gauge max |ΔE| 2.842170943040401e-14.

## Primary: per-instance gauge-average Nov3−QAOA E (n=8)

mean ΔE -0.3249 CI [-0.429, -0.249] median -0.2744 min -0.6500 max -0.2080.
Post-hoc F: mean Nov3 0.608 QAOA 0.489 (not selected).
Not an adoption gate. CI entirely <0 is a descriptive fact only.

## Identity-only (g=0)

mean ΔE -0.3236 CI [-0.427, -0.247].

## Each frozen gauge

- g=0: mean ΔE -0.3236 CI [-0.427, -0.247]
- g=1: mean ΔE -0.3269 CI [-0.433, -0.250]
- g=2: mean ΔE -0.3255 CI [-0.429, -0.250]
- g=3: mean ΔE -0.3236 CI [-0.428, -0.248]

## Per-instance range and worst gauge (largest ΔE, weakest Nov3 edge)

Mean Nov3 cross-gauge energy range 0.0055 (expressivity/labeling; SPSA seed shared).
Mean worst-gauge ΔE -0.3224.

- seed 10552: ΔE g=[-0.6438, -0.659, -0.6493, -0.6481] mean -0.6500 range 0.0152 worst g=0
- seed 10553: ΔE g=[-0.2463, -0.2476, -0.247, -0.2469] mean -0.2470 range 0.0013 worst g=0
- seed 10554: ΔE g=[-0.2999, -0.3001, -0.2928, -0.2934] mean -0.2965 range 0.0072 worst g=2
- seed 10555: ΔE g=[-0.3641, -0.3617, -0.364, -0.3617] mean -0.3629 range 0.0024 worst g=3
- seed 10556: ΔE g=[-0.2519, -0.2532, -0.2524, -0.2514] mean -0.2522 range 0.0018 worst g=3
- seed 10557: ΔE g=[-0.2085, -0.217, -0.2198, -0.2105] mean -0.2140 range 0.0112 worst g=0
- seed 10558: ΔE g=[-0.2061, -0.208, -0.2092, -0.2087] mean -0.2080 range 0.0031 worst g=0
- seed 10559: ΔE g=[-0.3683, -0.3689, -0.3698, -0.3682] mean -0.3688 range 0.0016 worst g=3

## QAOA sanity

max |E_g−E_0| 2.842e-14 (ok True, bar 1e-08). Implementation check, not assumed.

F is post hoc. No cherry-picked best gauge. Stop after this panel.

Fit workers 4, BLAS 1. Each completed ledger equals Q=1024. No hardware claim.

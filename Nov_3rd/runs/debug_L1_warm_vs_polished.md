# Nov_3rd debug — L=1 warm-only vs polished

_Asia/Shanghai 2026-11-03 15:25_

Motif: Nov1st-final X-sandwich, QAOA-warm shell SNAP, BinH b=3.
Warm = γ=γ_QAOA, φ=shell_phi_from_betas(β) (NO free polish).
Polished = params_best_L1.pt (n6×1000 LBFGS free γ+φ).

## Params

- QAOA β=[0.7718809843063354]
- warm γ=[-5.4505720138549805]
- warm φ=[[-0.0, -0.7718809843063354, -1.543761968612671, -2.315642833709717]]
- polished γ=[-5.1701483726501465]
- polished φ=[[-0.037413664162158966, -1.0219957828521729, -1.4764010906219482, -2.0954749584198]]
- ‖Δφ‖=0.3420
- Δγ=0.2804

## scale1000 Med / R vs QAOA L=1

| n | QAOA Med | warm Med | warm R | warm | polished Med | polished R | polished |
|---|----------|----------|--------|------|--------------|------------|----------|
| 6 | 9.779 | 9.779 | 1.000× | TIE | 9.502 | 0.972× | BEAT |
| 9 | 29.279 | 29.279 | 1.000× | TIE | 29.537 | 1.009× | TIE |
| 12 | 90.033 | 90.033 | 1.000× | TIE | 93.283 | 1.036× | UNDER |
| 15 | 289.892 | 289.892 | 1.000× | TIE | 318.811 | 1.100× | UNDER |

## Verdict

Root cause: n6×1000 free LBFGS polish on φ (shell) drifted off QAOA dual and hurt n12+ transfer — warm-only Motif = QAOA (R=1.000); Probe A (φ frozen) confirms φ drift was the damage (all TIE).

## Next fix

Probe A (freeze φ, train γ on n6) restored TIE everywhere (R≈1.00–1.01) but did not BEAT. Next fix: keep φ frozen at warm shell in the trainer; to BEAT QAOA try probe B sealed train_ns=(9,12) free/γ polish like E1 — n6-only γ polish is not enough to beat.


## Probe A — freeze φ, train γ only (n6×1000 LBFGS)

_Asia/Shanghai 2026-11-03 15:36_

Best seed=2; γ=[-5.20939826965332] (warm was [-5.4505720138549805])
φ frozen at warm shell.

| n | QAOA Med | probeA Med | R | verdict |
|---|----------|------------|---|---------|
| 6 | 9.779 | 9.894 | 1.012× | TIE |
| 9 | 29.279 | 29.410 | 1.004× | TIE |
| 12 | 90.033 | 90.119 | 1.001× | TIE |
| 15 | 289.892 | 290.541 | 1.002× | TIE |

Probe A (freeze φ, train γ on n6): all R≤1.02 — φ drift was the damage; keep φ frozen.


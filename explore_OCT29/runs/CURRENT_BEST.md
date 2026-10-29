# CURRENT BEST — explore_OCT29 (Asia/Shanghai)

_Updated 2026-10-29 ~18:08 CST_

## Fixed-angle / schedule champion

**Fourier q=3, L_train=6** on projector `W·SNAP₀(β)·W†`  
`runs/probe3_fourier_deep/modes_q3_Ltrain6.pt` (commit `d1399b4d`)

| deploy L | n15 Med/B90 | b_Med (9–18) |
|----------|-------------|--------------|
| 12 | 7.70/10.6 | ~1.34 |
| 20 | **5.99**/16.6 | **~1.24** |

## Probe5 seq champ (beats Fourier gate)

After raising promote gate to q3_Ltrain6, Fourier-warmed free-angle projector at **L=16** promoted:

| | n9 | n12 | n15 | b_Med |
|--|----|-----|-----|-------|
| **fourier_proj_L16** (free γ/β, Fourier warm) | **1.58/2.41** | **2.36/3.77** | **2.51/5.07** | **1.080** |
| Fourier q3 deploy L20 (schedule only) | — | — | 5.99/16.6 | 1.24 |

- Motif still `prep=W, layer=COST→PROJ` — depth + free angles, not a new mixer
- Gate: n15≲7.7 / b≲1.24; weak pre-gate champ archived as `champ_pre_fourier_gate.json`
- Loop alive: `scripts/probe5_seq_vocab.py` (stop: `touch runs/STOP_PROBE5`)

## Takeaway

Ceiling to beat for *novel* seq atoms is now ~**n15 Med 2.5 / b≈1.08** (L16 free-angle projector), not just Fourier 5.99. Mutants that only rearrange SNAP/D without matching this remain rejected.


## Fixed-γ deepen (2026-10-29) — negative

One γ (shared or per-instance) + deep L **does not scale**: n15 Med worsens past L≈8 (shared L8≈40 → L24≈686). Per-instance γ opt helps little. **Keep γ_ℓ / Fourier.** See `runs/fixed_gamma_deep/`.

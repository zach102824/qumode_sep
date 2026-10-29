# CURRENT BEST — explore_OCT29 (Asia/Shanghai)

_Updated 2026-10-29 ~20:30 CST_

## Promote gate (CHANGED — Opus 20:25)

**Do not promote on probe5 tiny-B val or on sealed test.**

- **val2** = full val packs: n15 **B=40**, n18 **B=16** (pack max; cannot hit B≥40 at n18 without regen).
- Gate on val2: n15 Med/B90, n15 L×Med, n18 Med/B90, leak, solved frac (`frac_inv_le_2`).
- **Test sealed** — look only at milestones.
- **Retired yardstick:** probe5 val b≈1.08 / n15 Med≈2.51 (selection bias + 1/p floor).

## Arm A reality check (probe6, bigger B)

| split | n15 Med/B90 | n18 Med/B90 | L×Med15 |
|-------|-------------|-------------|---------|
| val   | 2.60 / 5.91 | 6.06 / 75.1 | 41.5 |
| **test** | **6.33 / 24.0** (B=40) | **6.41 / 27.5** (B=16) | **101** |

## Fixed-angle / schedule champion (still)

**Fourier q=3, L_train=6** on projector `W·SNAP₀(β)·W†`  
`runs/probe3_fourier_deep/modes_q3_Ltrain6.pt`

| deploy L | n15 Med/B90 (old small-B val) | b_Med (9–18) |
|----------|-------------------------------|--------------|
| 20 | **5.99**/16.6 | **~1.24** |

## Probe5 seq champ (STOPPED — Opus 20:25)

Motif still `prep=W, layer=COST→PROJ` L=16 free angles — **no new motif in 197 trials**.

| | n9 | n12 | n15 | b_Med (val tiny-B) |
|--|----|-----|-----|--------------------|
| fourier_proj_L16 | 1.58/2.41 | 2.36/3.77 | 2.51/5.07 | 1.08 (retired) |

- Stopped via `runs/STOP_PROBE5`. Top configs archived in `runs/probe5_seq_vocab/top5_for_reval.json` for later val2 re-score.
- Do not restart probe5 SNAP rearrangements.

## Live / next

- **probe6** spectral mixer Arms B/C still running (Opus prior #1). Score under val2/test gate when done.
- **probe7** Fourier+α multi-depth distill **STARTED** (Opus 20:25 #2 pick). Stop: `touch runs/STOP_PROBE7`.

## Fixed-γ deepen — negative

One γ + deep L does not scale. Keep γ_ℓ / Fourier.

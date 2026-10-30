# CURRENT BEST — explore_OCT29 (Asia/Shanghai)

_Updated 2026-10-30 ~08:30 CST_

## Promote gate (Opus 20:25)

**Do not promote on probe5 tiny-B val or on sealed test.**

- **val2** = full val packs: n15 **B=40**, n18 **B=16** (pack max).
- Gate on val2: n15 Med/B90, n15 L×Med, n18 Med/B90, leak, solved frac (`frac_inv_le_2`).
- **Test sealed** — look only at milestones.
- **Retired yardstick:** probe5 val b≈1.08 / n15 Med≈2.51 (selection bias + 1/p floor).

---

## CHAMP — probe6 Arm C free shells (PROMOTED)

**Motif:** prep=`W_H`, layer `COST → W_H·SNAP(φ_{pc})·W_H†`, L=16  
φ_k = φ_{popcount(k)} for k=0..3 (4 shell phases/layer). Warm-started from PROJ.  
Params: `runs/probe6_spectral_mixer/arm_C_shell_params.pt`

| split | n9 Med/B90 | n12 | n15 | n18 | L×Med15 |
|-------|------------|-----|-----|-----|---------|
| val (probe6 B≤24/16) | 1.43/2.56 | 2.16/7.32 | 2.26/5.27 | 5.31/36.0 | 36.1 |
| **test** | **1.34/2.09** | **2.18/10.3** | **4.50/16.5** | **3.76/11.8** | **71.9** |

- test b_Med (n12/15/18) ≈ **1.10**; leak ≈ 0.
- vs Arm A PROJ on test: n15 **1.41×**, n18 **1.70×** (promote ok).
- Note: probe6 val n15 used B=16; when gating newcomers, re-score Arm C on **val2 B=40**.

### Arm A (prior PROJ) — demoted

| split | n15 Med/B90 | n18 Med/B90 | L×Med15 |
|-------|-------------|-------------|---------|
| val | 2.60 / 5.91 | 6.06 / 75.1 | 41.5 |
| test | 6.33 / 24.0 | 6.41 / 27.5 | 101 |

### Arm B (X-only spectral) — dead

test n15 Med ~5e4. Do not revive.

---

## probe7 Fourier+α — NEGATIVE (do not restart α recipe)

`runs/probe7_fourier_alpha/`

- Phase1 LS distill q=8 OK-ish on val2 (~4.08 at distill alone, earlier log).
- Phase2 multi-L + α **ruined** it: test n15 L24 Med **~158** / B90 ~1075.
- Do **not** restart the probe7 α multi-depth recipe.

---

## Fixed-angle / schedule reference (still useful)

**Fourier q=3, L_train=6** on old projector — `runs/probe3_fourier_deep/modes_q3_Ltrain6.pt`  
deploy L20: old small-B val n15 Med **5.99**, b≈1.24. Below Arm C test bar.

## Probe5 seq — STOPPED

Motif rearrangements exhausted (~197 trials). Do not restart SNAP/PROJ shuffle.  
Top configs: `runs/probe5_seq_vocab/top5_for_reval.json`.

## Fixed-γ deepen — negative

One γ + deep L does not scale.

---

## Live / next

- **probe8_span_split** (Opus #3): COST γ_ℓ^(s) by clause span 2/3/4; mixer = Arm C shells; warm from Arm C. Train {9,12}. Gate val2. Stop: `touch runs/STOP_PROBE8`.
- Next after probe8: Opus #4 pair-projector (if span-split soft/negative).
- No jobs from probe6/probe7 — both DONE.

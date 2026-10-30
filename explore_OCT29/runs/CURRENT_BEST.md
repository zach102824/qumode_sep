# CURRENT BEST — explore_OCT29 (Asia/Shanghai)

_Updated 2026-10-30 ~09:10 CST_

## Promote gate (Opus 20:25)

**Do not promote on probe5 tiny-B val or on sealed test.**

- **val2** = full val packs: n15 **B=40**, n18 **B=16** (pack max).
- Gate on val2: n15 Med/B90, n15 L×Med, n18 Med/B90, leak, solved frac (`frac_inv_le_2`).
- **Test sealed** — look only at milestones.
- **Retired yardstick:** probe5 val b≈1.08 / n15 Med≈2.51 (selection bias + 1/p floor).

---

## CHAMP — probe8 span-split on Arm C (PROMOTED soft on val2)

**Motif:** COST `γ_ℓ^(s)` by clause span s∈{2,3,4}; mixer = Arm C `W_H·SNAP(φ_pc)·W_H†`, L=16.  
Warm from Arm C. Params: `runs/probe8_span_split/span_params.pt`

| split | n9 Med/B90 | n12 | n15 | n18 | L×Med15 |
|-------|------------|-----|-----|-----|---------|
| **val2** | **1.44/2.33** | **2.12/6.60** | **2.75/10.0** | **4.81/29.6** | **44.0** |
| **test** (milestone) | — | — | **4.21/16.0** | **3.16/10.7** | **67.3** |

- Promote vs Arm C val2: n15 **×1.07**, n18 **×1.10** (ok soft).
- **Caveat (Opus 09:03):** learned γ ratios ≈1.0 — gain may be optimizer/LR effect, not span physics. Still real on test (n15 ×1.07, n18 ×1.19 vs Arm C prior test).

### Prior — probe6 Arm C free shells (demoted to #2)

**Motif:** prep=`W_H`, layer `COST → W_H·SNAP(φ_{pc})·W_H†`, L=16  
Params: `runs/probe6_spectral_mixer/arm_C_shell_params.pt`

| split | n15 Med/B90 | n18 Med/B90 |
|-------|-------------|-------------|
| val2 (probe8 re-score) | 2.949 / 11.3 | 5.306 / 36.0 |
| test (probe6) | 4.50 / 16.5 | 3.76 / 11.8 |

### Arm C B=1000 scale ladder (live — n18 running)

| n | Med | B90 | B |
|---|-----|-----|---|
| 9 | 1.441 | 3.314 | 1000 |
| 12 | 1.991 | 6.252 | 1000 |
| 15 | 2.921 | 10.959 | 1000 |
| 18 | (running ~0.11/s) | — | 1000 |

b_Med≈**1.125** (ns 9–15); b_B90≈**1.221**. Do **not** kill n18.

### Arm A (prior PROJ) — demoted

test n15 Med 6.33 / B90 24.0

### Arm B (X-only spectral) — dead

test n15 Med ~5e4. Do not revive.

---

## probe7 Fourier+α — NEGATIVE (do not restart α recipe)

Phase2 multi-L + α ruined distill. Do **not** restart.

---

## Probe5 seq — STOPPED

Motif rearrangements exhausted. Do not restart SNAP/PROJ shuffle.

---

## Live / next

- **probe10_tail_loss** LIVE (Opus 11:08 #1): τ∈{1,2,plain} log-mean-exp loss, warm from span; nsol1~50% batches; 200 Adam + LBFGS. Stop: `touch runs/STOP_PROBE10`. Log: `runs/probe10_tail_loss/run.log`.
- Early tau1 val2 n15 (mid-run): Med≈2.68 / B90≈9.09 vs span 2.75/10.0 — confirm after all arms + n18/scale1000.
- probe9 pair-projector NEGATIVE (promote False). Do not restart.
- Arm C B=1000: n9/12/15 done (b_Med≈1.125, b_B90≈1.221); n18 killed at 700/1000 Med≈4.481.
- compile_W_H stopped (8-pair F8≈0.987). Opus: skip Gray-X E3; prefer compile learned mixers (probe11) after probe10.
- Do not revive GNN / probe5 / probe7-α / Arm B / pair-projector.


---

## Soft-promoted — projector-gadget Uc (BinH) — 2026-10-30 ~19:30 CST

Motif: clause-projector cost + Arm C shell mixer, BinH W. Params: `runs/gadget_Uc/unsat_params.pt`

| n | Med | B90 | B |
|---|-----|-----|---|
| 9 | 1.445 | 3.222 | 1000 |
| 12 | 1.926 | 6.148 | 1000 |
| 15 | 2.820 | 10.063 | 1000 |
| 18 | 4.329 | 24.342 | 1000 |

Matches/beats Arm C B1000 on n9–18. Parity/stencil retired. Fourier deepen discarded.

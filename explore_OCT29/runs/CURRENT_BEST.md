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


## Projector-gadget champ — train{6,9,12} (2026-10-30 ~19:38 CST)

Params: `runs/gadget_Uc/unsat_params.pt` (promoted from train6912).

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.406 | 3.288 |
| 12 | 1.864 | 6.106 |
| 15 | 2.754 | 10.244 |
| 18 | 4.194 | 24.302 |

GRAPE library: `runs/gadget_Uc/grape_library/LIBRARY.md` (26 local projectors; SNAP exact).


### Freedom ladder (2026-10-30) — negative

span_w / eps_delta: absolute Med dips but miss promote gate on n18. Champ unchanged.


## Champ UPDATE — projector L=20 (2026-10-30 ~19:53 CST)


| n | Med | B90 |
|---|-----|-----|
| 9 | 1.261 | 2.657 |
| 12 | 1.590 | 4.760 |
| 15 | 2.168 | 8.296 |
| 18 | 3.170 | 19.916 |

b_Med≈1.1079, b_B90≈1.2460. Params: `runs/gadget_Uc/unsat_params.pt`.


## Champ UPDATE — projector **L=24** (2026-10-30 ~19:59 CST)


| n | Med | B90 |
|---|-----|-----|
| 9 | 1.185 | 2.245 |
| 12 | 1.392 | 3.865 |
| 15 | 1.820 | 7.283 |
| 18 | 2.492 | 17.005 |

b_Med≈1.0868, b_B90≈1.2506.


## Champ UPDATE — projector **L=32** (2026-10-30 ~20:18 CST)


| n | Med | B90 |
|---|-----|-----|
| 9 | 1.109 | 1.828 |
| 12 | 1.258 | 3.013 |
| 15 | 1.537 | 5.767 |
| 18 | 1.995 | 14.117 |

b_Med≈1.0676. Span-split discarded.

---

## Locked projector LBFGS L32 (unchanged 2026-10-31)

Med/B90 n9–18: **1.092/1.776, 1.197/2.843, 1.418/5.505, 1.752/11.353** (b_Med≈1.054).

renames_Hc wave (`runs/gadget_Uc/renames_Hc/`): soft-n18 + SNAP φ noise on **retrain seed** (params.pt was never in git). Soft-n18 keep vs retrain only — **not** promoted over locked. WalkSAT scale1000 filled. Still `Uc=exp(-iγ Hc)`.

---

## Champ UPDATE — soft-n18 from locked LBFGS (2026-10-31 ~02:33 CST)

**CRITICAL FIX:** earlier soft-n18 (~n18 Med 10) was on bad retrain — INVALID.

Locked seed0 verified scale1000 = sealed (Δ~0). Soft-n18 `t9121518_w1` **PROMOTED**:

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.098 | 1.719 |
| 12 | 1.197 | 2.779 |
| 15 | 1.395 | 5.460 |
| 18 | 1.709 | 11.107 |

b_Med≈1.0506. Params: `unsat_params.pt`. Backup: `unsat_params_pre_soft18_backup.pt`.
n18 val in soft pool (not pure zero-shot); scale1000 holdout OK. Still exp(-iγ Hc).

---

## Champ UPDATE — L36 pad (2026-10-31 ~02:50 CST)

Interp soft-n18 L32→36, no retrain. Med/B90:

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.072 | 1.613 |
| 12 | 1.156 | 2.532 |
| 15 | 1.312 | 4.918 |
| 18 | 1.571 | 10.312 |

b_Med≈1.0434. Still exp(-iγ Hc). Multi-seed LBFGS on L32: no promote.

---

## Champ UPDATE — L40 pad (2026-10-31 ~02:53 CST)

| n | Med | B90 |
|---|-----|-----|
| 9 | 1.058 | 1.532 |
| 12 | 1.129 | 2.367 |
| 15 | 1.263 | 4.563 |
| 18 | 1.467 | 10.012 |

b_Med≈1.0371. Interp L36→40, no retrain. Still exp(-iγ Hc).

---

## Champ UPDATE — L44 pad (2026-10-31 03:09 CST)

Med/B90 n9–18: 1.051/1.473, 1.113/2.263, 1.233/4.243, 1.409/9.645
b_Med≈1.0333. L44 pad yes; soft18=no. L48 skipped. Still exp(-iγ Hc).

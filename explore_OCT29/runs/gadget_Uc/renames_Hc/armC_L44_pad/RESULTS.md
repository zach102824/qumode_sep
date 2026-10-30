# Fair ArmC pad L16→L44 — scale1000

_Generated 2026-10-31 03:59:52 +0800 (Asia/Shanghai)_

Honesty: ArmC = densified `exp(-iγ Hc)` in W_H frame; pad = linear interp of shell angles only (same recipe as projector depth pad). **Fair same-L baseline**, not a promote.

**L:** 44 (from 16)  **B:** 1000  **nf:** 16
**Params:** `/workspace/qumode_sep_fresh/explore_OCT29/runs/probe6_spectral_mixer/arm_C_shell_params.pt` → `/workspace/qumode_sep_fresh/explore_OCT29/runs/gadget_Uc/renames_Hc/armC_L44_pad/arm_C_shell_params_L44_pad.pt`

## Ladder

| n | ArmC L44 Med/B90 | ArmC L16 | champ L44 | L×Med | wall_min |
|---|------------------|----------|-----------|-------|----------|
| 9 | **1.162/1.754** | 1.441/3.314 | 1.051/1.473 | 51.1 | 0.1 |
| 12 | **1.353/3.291** | 1.991/6.252 | 1.113/2.263 | 59.5 | 0.5 |
| 15 | **1.728/6.399** | 2.921/10.959 | 1.233/4.243 | 76.0 | 5.4 |
| 18 | **2.160/20.294** | 4.573/27.810 | 1.409/9.645 | 95.0 | 18.6 |

**n18 B=100** (cheap; L16 n18 B=1000 was 106 min → L44 full ≈5h deferred).

## b fits

- **9_18**: b_Med≈**1.0726**, b_B90≈**1.3061**
- **9_15**: b_Med≈**1.0683**, b_B90≈**1.2408**
- **12_18**: b_Med≈**1.0811**, b_B90≈**1.3542**
- **15_18**: b_Med≈**1.0772**, b_B90≈**1.4692**

## vs ArmC L16 (pad hurt?)

- n9: Med ratio L44/L16 = 0.806 (improved)
- n12: Med ratio L44/L16 = 0.679 (improved)
- n15: Med ratio L44/L16 = 0.591 (improved)
- n18: Med ratio L44/L16 = 0.472 (improved)

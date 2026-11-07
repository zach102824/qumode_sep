# Opus Idea 2 — cavity-locality-graded γ (graded H_c)

_L=2 primary, train{6,9} only, CORE {6,9,12,15}, B=100 smoke. 2026-11-04._

**Formula.** \(U_c^X = W\cdot\exp(-i[\gamma_{\ell,\mathrm{in}} E_{\mathrm{in}} + \gamma_{\ell,x} E_x])\cdot W\)

- \(E_{\mathrm{in}}\): Walsh / X-strings whose support lies in one 3-bit cavity (plus DC).
- \(E_x\): strings that touch **≥2** cavities. **Two classes only** — not split by exact span 1/2/3/4.
- Warm: \(\gamma_{\mathrm{in}}=\gamma_x=\gamma_{\mathrm{QAOA}}\). Hamming-shell SNAP \(\varphi_k=-\beta\cdot k\), BinH \(W=H^{\otimes 3}\).
- Label: **graded H_c** (reweight of the existing X-sandwich cost). Not a non-\(H_c\) arm.
- Gate: **BEAT** iff \(R=\mathrm{Med}/\mathrm{Med}_{\mathrm{QAOA}}\le 0.98\) on CORE. Smoke R uses **same-B** QAOA Med.

Script: `Nov_3rd/scripts/train69_ucx_idea2_cavity.py`.

## Med / R table (same-B B=100)

Warm identity (γ_in=γ_x=γ_QAOA, no train): **R = 1.0000** on all four CORE sizes (locked motif recovered).

### Primary: free γ_in, γ_x, c1–c3 (QAOA-warm)

| seed | n6 R | n9 R | n12 R | n15 R | ALL_CORE_BEAT |
|------|------|------|-------|-------|---------------|
| 0 | **0.9374** BEAT | **0.9793** BEAT | **0.9705** BEAT | 1.0290 UNDER | False |
| 1 | **0.9694** BEAT | 0.9810 TIE | 0.9881 TIE | 1.0105 TIE | False |
| 2 | **0.9658** BEAT | **0.9646** BEAT | **0.9708** BEAT | 0.9832 TIE | False |

Best knee among n6&n9-BEAT: **seed 2** — n12 BEAT 0.9708, n15 TIE **0.9832** (still above 0.98).

### Seed 2 (best transfer) detail

| n | QAOA Med | Med | R | verdict | E_x frac |
|---|----------|-----|---|---------|----------|
| 6 | 5.0038 | 4.8326 | **0.9658** | BEAT | 0.550 |
| 9 | 11.2179 | 10.8213 | **0.9646** | BEAT | 0.623 |
| 12 | 33.2021 | 32.2314 | **0.9708** | BEAT | 0.665 |
| 15 | 71.6819 | 70.4755 | **0.9832** | TIE | 0.671 |

### Seed 0 detail

| n | QAOA Med | Med | R | verdict | E_x frac |
|---|----------|-----|---|---------|----------|
| 6 | 5.0038 | 4.6905 | **0.9374** | BEAT | 0.550 |
| 9 | 11.2179 | 10.9854 | **0.9793** | BEAT | 0.623 |
| 12 | 33.2021 | 32.2243 | **0.9705** | BEAT | 0.665 |
| 15 | 71.6819 | 73.7615 | **1.0290** | UNDER | 0.671 |

### Freeze-SNAP ablation (only γ_in, γ_x; seed 0)

| n | R | verdict |
|---|---|--------|
| 6 | 0.9936 | TIE |
| 9 | 0.9898 | TIE |
| 12 | **0.9733** | BEAT |
| 15 | 1.0133 | TIE |

SNAP polish is doing most of the n=6/9 work; the γ split alone is a mild n12 bump and still loses n15.

## E_x share (Walsh energy, DC excluded)

| n | cavities | E_x frac | span-1 | span-2 | span-3 | span-4 | span-5 |
|---|----------|----------|--------|--------|--------|--------|--------|
| 6 | 2 | 0.550 | 0.450 | 0.550 | — | — | — |
| 9 | 3 | 0.623 | 0.377 | 0.491 | 0.132 | — | — |
| 12 | 4 | 0.665 | 0.335 | 0.476 | 0.178 | 0.011 | — |
| 15 | 5 | 0.671 | 0.329 | 0.455 | 0.197 | 0.020 | 0.000 |

E_x grows with n, as predicted. Span-5 is empty (4-SAT Walsh weight ≤4). Span-4 strings exist at eval n=12/15 but **never in train{6,9}**; they still share γ_x with span-2/3 (by design).

## γ (primary)

QAOA warm: γ = (−4.196, −6.868).

| seed | γ_in | γ_x | rel ‖γ_in−γ_x‖ | unused? |
|------|------|-----|----------------|---------|
| 0 | (−4.385, −6.700) | (−4.071, −7.060) | 0.061 | no |
| 1 | (−4.339, −6.688) | (−4.113, −7.058) | 0.054 | no |
| 2 | (−4.272, −6.781) | (−4.133, −6.961) | 0.029 | no |
| freeze SNAP s0 | (−5.387, −4.783) | (−4.035, −7.239) | 0.374 | no |

Extra freedom is used (γ_in ≠ γ_x). Freeze-SNAP drives a much larger split because SNAP cannot absorb local work.

## Kill / pass

- **Pass (R≤0.98 all CORE):** **FAIL.** No seed is ALL_CORE_BEAT. Best n15 = 0.9832 TIE.
- **Kill if n=12/15 no better than ~0.986:** n12 *is* better (0.971 BEAT on s0/s2). n15 is **not** (0.983–1.029). Matches the L=2 near-miss knee.
- **Kill if γ_in, γ_x collapse equal:** **no** (rel 3–6%).
- **Kill if R steadily worse with n (n15>n12>n9):** seed 1 yes (0.981→0.988→1.011); seed 2 yes (0.965→0.971→0.983); seed 0 n15 UNDER after n12 BEAT. Predicted failure mode: γ_x fit on 2–3 cavities is too strong at n=15 (E_x 0.55→0.67).

**Verdict (B=100 smoke):** idea 2 is a real but insufficient L=2 lever. Honest graded H_c, identity-warm, two-class split as specified. Does not close ALL_CORE. Do not promote. If idea 1 also shows a signal, the 4×2 (degree × in/across) grid is the writeup follow-up — not a deeper span split.

## Files

- `params_L2_seed{0,1,2}.pt`, `eval_L2_seed{0,1,2}.json/.md`
- `warm_only/` identity R=1
- `freeze_snap/` SNAP-frozen ablation
- `logs/`

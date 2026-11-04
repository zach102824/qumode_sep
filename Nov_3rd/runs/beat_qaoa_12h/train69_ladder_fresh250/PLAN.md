# PLAN — power_p Hamming-shell SNAP vs QAOA (box)

_Asia/Shanghai 2026-11-04 ~12:13 — Zach t2633–t2637 go_

## Motif (locked)
\[
|0\rangle \to [U_c^{X}\to\mathrm{Hamming\text{-}shell\,SNAP}]^{L} \to W \to \mathrm{Fock},
\quad U_c^{X}=W D W,\quad W=H^{\otimes 3}.
\]
SNAP \(\phi_k=c_1 k+c_2 k^2+c_3 k^3\).

**power_p:** \(a_n=(8/(n-1))^p\) scales \(c_2,c_3\); learn \(p\) with \(\gamma,c_1{-}c_3\).

## Fresh train (SNAP **and** QAOA)
- New packs: **250 n6 + 250 n9**, disjoint from scale1000 eval
- Each step: **batch 32 n6 + 32 n9**
- Loss weights: **w6=1, w9=2** (SNAP); QAOA trained on the same n6/n9 mix
- Paper QAOA init: \(\beta=0.01\), \(\gamma=-0.01\), nstart=1
- Do **not** reuse old paper-n6-only QAOA angles or old L=16 champ without retrain on this fresh set

## Phase 1 — CORE first
- \(L\in\{1,4,8,12,16,20,24,28,32\}\) small→large
- Per L: train QAOA fresh → train power_p (seed0; more seeds if needed) → eval **B=1000**
- **Eval n:** \(\{6,9,12,15\}\) only
- Report **Med** and **B90** for SNAP and same-L QAOA (**no R**)
- Gate on raw Med: **BEAT** if Med \(<\) Med\(_\mathrm{QAOA}\); **TIE** if equal; **UNDER** if greater
- Same BEAT/TIE/UNDER mark for B90 vs QAOA B90
- Deliverable: RESULTS table rows=L, cols=n6/9/12/15 Med & B90 (SNAP + QAOA) + marks

## Phase 2 — after Phase 1 finishes all L
- Add **n=18** then **n=21** eval (same trained angles; no retrain unless asked)
- Extend RESULTS with n18/n21 Med & B90

## Out of scope
Walsh / cavity / clause Idea arms, Opus loop, clouds (unless asked)

## Run location
`Nov_3rd/runs/beat_qaoa_12h/train69_ladder_fresh250/`

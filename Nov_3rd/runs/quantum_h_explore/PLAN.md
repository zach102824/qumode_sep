# PLAN — quantum H explore (keep Nov_3rd gate sequence)

_Asia/Singapore 2026-11-04 ~21:50 — Zach t2710–t2713: nonstop explore_

## Goal
Find a **quantum** problem where the **current** BinH \(U_c^X\) ↔ Hamming-shell SNAP
motif works **very well**, so that GRAPE of \(\mathrm{e}^{-i H_c}\) is justified
(structured cost pulse ≠ full answer). Keep beating Fourier QAOA on the **same** \(H\).

## Locked constraints (pass all or kill)

| # | lock | note |
|---|------|------|
| G | Gate sequence | \(U_c^X = W\,e^{-i\gamma H_c}\,W\), \(W=H^{\otimes 3}\); Hamming-shell SNAP on popcount/Fock |
| Q | Quantum GS | GS not a classical bitstring; needs noncommuting terms |
| R | Readout | **Z-hist + X-hist** enough for \(\langle H\rangle\) (no YY / XX correlators required) |
| U | \(U_c^X\) justified | \(H_c\) diagonal piece GRAPE can apply; full \(H\) still hard |
| T | Transfer | train small \(n\in\{6,9\}\) → larger \(n\); smoke on n6/n9 OK |
| B | Beat target | Fourier QAOA on **same** \(H\); metric \(\langle H\rangle/n\) or gap to \(E_0\) at small \(n\) |
| C | Structure | Prefer cross-cavity **3-local** matching Max-E / clause packing |

## Non-goals (this folder)
- Do **not** launch heavy trains / cloud agents.
- Do **not** disrupt phase2 n21 ladder (`run_train69_ladder_fresh250_phase2.py`, PID ~143971).
- Do **not** change `Nov_3rd/settings.py` locks or motif code for the classical Max-E ladder.

## Code reuse hooks (existing Nov_3rd)
- **Diag cost table:** packs expose `*_E` shape `(B, 2^n)` = unsat counts (= classical \(H_c\) on bitstrings).
  - n6: `data/scale1000/data_n6.npz` → `scale1000_E (1000, 64)`, `m=60`, clauses `cls_v/cls_s (1000,60,4)`.
  - n9: `data_n9.npz` → `scale1000_E (1000, 512)`, `m=89`.
- **Apply \(U_c^X\):** `Nov_3rd/ansatz.py` `Nov3rdAnsatz.forward_nov3rd` / `cost_Cn` / Walsh-degree `graded_cost_diag`.
- **QAOA fair ref:** `ensure_qaoa.py` + `qaoa_angles/L{L}/` (paper init); for quantum \(H\) need a **new** QAOA cost = full \(H\) (or at least train angles vs \(\langle H\rangle\)), not classical unsat Med.
- **Smoke path:** build dense \(H\) at n6/n9 (`2^6=64`, `2^9=512`), `numpy.linalg.eigh` / sparse Lanczos for \(E_0\); estimate \(\langle H\rangle\) from circuit Z-hist + per-qubit X-hist.

## Deliverables in this folder
| file | content |
|------|---------|
| `SHORTLIST.md` | ≥6 families ranked vs locks; TOP 3 with formulas / λ / win story / kill |
| `SMOKE.md` | L∈{2,16}, n∈{6,9}, B small, seeds, Walsh/power_p vs Fourier QAOA, exact \(E_0\) |
| `PLAN.md` | this file |

## Decision rule after smoke
1. Pick \(\lambda\) (or \(\alpha\)) where GS is clearly quantum (overlap with any bitstring \(\ll 1\), or \(\langle Z\rangle\) not saturating).
2. Method **BEAT** if median \(\langle H\rangle\) (or \(\langle H\rangle/n\)) ≤ \(0.98\times\) same-L Fourier QAOA on held-out B, **and** gap-to-\(E_0\) not worse than QAOA at n6/n9.
3. Kill family if: needs extra correlators; GRAPE of \(H_c\) alone already finds GS; or SNAP never beats QAOA across λ grid.
4. Winner → small train{6,9} recipe freeze → only then touch ladder / new run dir. **Never** on n21 PID.

## Status
- Docs written 2026-11-04 ~21:50 Asia/Singapore.
- No smoke jobs launched from this explore pass.
- n21 phase2 ladder left running.

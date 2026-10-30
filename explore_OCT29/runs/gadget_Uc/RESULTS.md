# Gadget-Uc — projector-gadget family (SOFT-PROMOTED)

_Generated 2026-10-30 19:33:57 +0800 (Asia/Shanghai)_

Plan: `docs/PLAN_gadget_Uc.md`. **BinH** \(W=H^{\otimes 3}\), mixer Arm C shell `W·SNAP(φ_pc)·W†`, L=16.

## Soft promote

- **KEEP:** per-clause **projector gadgets** `U_c=∏_c[I+(e^{-iγ}-1)C_c]` (= `diag(e^{-iγ E})` algebraically; GRAPE targets clause ops / BinH X-projectors — not densified global `H_cost` as pulse target).
- **DROP from further runs:** XOR parity `∏X` strings; instance-blind `stencil_x`.
- Params: `unsat_params.pt` (train `{9,12}`).

## scale1000 Med/B90 (B=1000) — projector vs Arm C

| n | projector (unsat) | Arm C ref | ratio ArmC/proj Med |
|---|-------------------|-----------|---------------------|
| 9 | 1.445/3.222 | 1.441/3.314 | ×0.997 |
| 12 | 1.926/6.148 | 1.991/6.252 | ×1.034 |
| 15 | 2.820/10.063 | 2.921/10.959 | ×1.036 |
| 18 | 4.329/24.342 | 4.573/27.810 | ×1.056 |

- n18 wall ≈ 8.2 min; frac_inv≤2 = 0.092; leak≈7.68e-06

## Retired arms (do not restart)

| arm | scale1000 n15 Med/B90 | note |
|-----|----------------------|------|
| parity XOR | 13150/87786 | wrong cost |
| stencil_x | 16384/32768 | no SAT signal |

## Fourier-γ deepen nudge — DISCARD

See `fourier_deepen/RESULTS.md`. q=3 Fourier γ+φ train L16 deploy L16/20:
- L16 scale n9 Med **1.843** > projector **1.445**
- L20 test n15 Med **5.102** > projector test **4.167**
- **keep=False** — stick with free γ/φ `unsat_params.pt`.

## Operator reminder

- `∏_c [I+(e^{-iγ}-1)C_c] = diag(e^{-iγ E_unsat})` exact.
- BinH: each `C_c → C_c^X` (X-basis projector gadget).
- Single-Pauli XXX ≠ clause projector.

Artifacts: `unsat_params.pt`, `scale1000_n18.json`, `fourier_deepen/`, `run.log`, `run_unsat.log`, `eval_n18.log`.


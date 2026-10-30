# Gadget-Uc explore — X-gadget cost + Arm C shell mixer (BinH)

_Generated 2026-10-30 19:19:03 +0800 (Asia/Shanghai)_

Plan: `explore_OCT29/docs/PLAN_gadget_Uc.md`.

**Locked:** BinH \(W=H^{\otimes 3}\), mixer = Arm C shell `W·SNAP(φ_pc)·W†`, train `{9,12}` L=16.

## What is instance-blind vs aware

| Piece | Blind? |
|-------|--------|
| Gadget family / SNAP controls | YES (GRAPE once) |
| Which clause projectors / X-strings | NO (classical schedule) |
| `stencil_x` entire \(U_c\) | YES fully |
| `γ_ℓ`, `φ_pc` | shared train |

## Operator checks

- BinH: `W† exp(-iγ ZZZ) W = exp(-iγ XXX)` fro ~1e-15 PASS
- stencil: `∏ exp(-iγ X_j) = W† exp(-iγ ∑ Z_j) W` fro ~1e-15 PASS
- Clause projector: `∏_c [I+(e^{-iγ}-1)C_c] = diag(e^{-iγ E_unsat})` exact. BinH ⇒ X-projector gadgets.

## scale1000 Med/B90 (B=1000) — main gate

| n | parity (XOR X-string) | stencil_x (blind) | unsat / projector-gadget | Arm C ref |
|---|----------------------|-------------------|--------------------------|-----------|
| 9 | 309.767/2158.960 | 256.000/512.000 | 1.445/3.222 | 1.441/3.314 |
| 12 | 1912.569/11831.688 | 2048.001/4096.005 | 1.926/6.148 | 1.991/6.252 |
| 15 | 13150.490/87785.854 | 16383.996/32768.035 | 2.820/10.063 | 2.921/10.959 |

### unsat vs Arm C (scale1000 Med ratio ArmC/unsat; >1 means unsat better)

| n | ratio |
|---|-------|
| 9 | ×0.997 |
| 12 | ×1.034 |
| 15 | ×1.036 |

## test ladder (small packs)

| n | parity | stencil_x | unsat |
|---|--------|-----------|-------|
| 9 | 247.592/598.573 | 256.000/512.000 | 1.364/2.012 |
| 12 | 2084.425/10576.339 | 2048.002/4096.004 | 2.137/8.499 |
| 15 | 22242.663/109403.417 | 32767.982/32768.051 | 4.167/15.725 |
| 18 | 75755.387/377338.750 | 58982.453/262144.016 | 3.263/10.675 |

## Promote verdict

- **parity**: FAIL (Med hundreds–10k). Single Pauli `∏X` per clause is Max-XOR/parity, not 4-SAT unsat.
- **stencil_x**: FAIL (Med≈`2^{n-1}`). Instance-blind cost carries no SAT signal.
- **unsat / clause-projector gadgets**: **PROMOTE soft** — competitive with Arm C B1000 on n9/12/15 (BinH). This is the GRAPE-correct exact cost: per-clause ops `I+(e^{-iγ}-1)C_c` (Z) or conjugated X-projectors — **not** densifying global `H_cost` as a GRAPE pulse target; instance only picks which projectors.

Single-Pauli XXX gadgets do **not** replace unsat projectors. Keep projector-gadget family for GRAPE; drop XOR-parity and blind stencil for SAT.

Artifacts: `parity_params.pt`, `stencil_x_params.pt`, `unsat_params.pt`, `results.json`, `run.log`, `run_unsat.log`.


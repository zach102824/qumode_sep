# GRAPE library FREEZE — projector-Uc (BinH)

_Generated 2026-10-30 20:32:36 +0800 (Asia/Shanghai)_

## Status: FREEZE classical search on angles/depth

Champ: L=32 projector-Uc soft-n15 (`t91215`), scale1000 Med/B90:
n9 1.100/1.759, n12 1.231/2.919,
n15 1.483/5.847, n18 1.884/13.704.
b_Med≈1.0618. Params: `runs/gadget_Uc/unsat_params.pt`.

## Frozen GRAPE operator library

1. **26 local clause projectors** on 3-bit cavity (k∈{1,2,3}, signs) — `LIBRARY.md`.
2. **Two-cavity** span-2: \(C=C_a\otimes C_b\) for (1,3)/(2,2); span-3/4 = more factors.
3. Unitary \(U(\gamma)=I+(e^{-i\gamma}-1)C\) — fixed C, only γ scales.
4. **Exact compile (no instance densify):**
   - Z-frame: \(U_Z=\mathrm{SNAP}(\theta)\) with \(\theta_f=\gamma\) on Fock support of \(C_Z\).
   - BinH X-frame: \(U_X=W^\dagger\mathrm{SNAP}(\theta)W\), \(W=H^{\otimes 3}\).
5. Shared schedule: \(\gamma_\ell\), shell \(\phi_{pc}\) (L=32); instance → which library ops.

## Explicitly NOT a GRAPE target

Densified instance `diag(E_unsat)` as a pulse optimization target.

## Next hardware-facing work

Two-cavity GRAPE/SNAP fidelity for \(C_a\otimes C_b\); pulse reuse across γ via time-scale / SNAP phase.

## Done (sim, not pulses)

Two-cavity factor/fidelity table: `COMPILE_TWOCAV.md` (2026-10-30). Kron of locals fails; joint SNAP exact.

## Offline compile (done)

See `COMPILE_OFFLINE.md` — 26 SNAP masks Fid=1; wall vs densify n12/15/18.

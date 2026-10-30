# GRAPE library RESULTS

_Generated 2026-10-30 19:36:14 +0800 (Asia/Shanghai)_

- Local projector patterns: **26** (k1+k2+k3)
- Exact SNAP / W†SNAP W identities: **PASS**
- Full write-up: `LIBRARY.md`

- Mean Fid SNAP·D fit to UX: **0.9022**
- Mean Fid D·SNAP·D fit to UX: **0.9343**

## Two-cavity compile (sim)

_Updated 2026-10-30 20:39:34 +0800 (CST)_

- Factorization: \(U(\gamma)=I+(e^{-i\gamma}-1)C_a\otimes C_b\) **does not** factor as \(U_a\otimes U_b\) (fid≈0.774 for projector-unitaries).
- Exact: joint Fock SNAP / BinH-conjugated joint SNAP.
- Local SNAP⊗SNAP approximate only; BS sandwich ≈ same as local kron (no AND phase).
- Full write-up: `COMPILE_TWOCAV.md` (+ `compile_twocav.json`).

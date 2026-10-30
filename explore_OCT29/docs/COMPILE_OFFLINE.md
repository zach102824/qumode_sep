# Offline GRAPE library compile (once)

_Generated 2026-10-31 00:02:23 +0800 (Asia/Shanghai)_

Not a Med chase. Documents **compile-once** of the frozen 26 local
projectors as exact SNAP / W†SNAP W, vs densifying full Arm C `Uc` per instance.

## Exact compile (26 locals)

- Patterns: **26** (k1=6, k2=12, k3=8).
- Offline wall: **0.0133 s** (enumerate + Fid checks).
- Fid SNAP↔UZ min: **1.000000000000000**
- Fid W†SNAP W↔UX min: **0.999999999999998**
- max BinH conj fro: **2.006e-15**
- All Fid ≥ 1−1e−12: **True** (exact within numerics).

Rule: store `snap_mask` (0/1 on Fock 0..7); at pulse time
`θ_f = γ_ℓ · snap_mask[f]`, then SNAP(θ) or W†SNAP(θ)W. **No instance densify.**

## γ-reuse (champ soft-n15 L32)

- Params: `runs/gadget_Uc/unsat_params.pt` L=32 cost_mode=unsat
- γ range: [0.4663, 6.6646] mean=4.2008
- Reuse: θ_f(γ) = γ * snap_mask[f]  (same 26 masks; only γ_ℓ from champ)

## Two-cavity (note only)

Span-2 clauses need `C=C_a⊗C_b`. Local SNAP⊗SNAP / Ua⊗Ub **fail** AND-phase
(Fid≈0.77). **Joint SNAP** on 8×8 is exact. See `COMPILE_TWOCAV.md`.
Offline library still stores the two local masks; joint pulse is a separate
compile target (not densified E).

## Wall-clock: library schedule vs densify E (Arm C Uc)

Densify = `fast_energies` → length-2^n E, then `exp(-iγ E)` per layer.
Library = clause→local `snap_mask` lookup + `γ·mask` (8 floats), **no 2^n alloc**.

| n | B | densify build E (s) | densify phase L32 (s) | lib schedule L32 (s) | densify/inst ms | lib/inst ms | speedup (phase) |
|---|---|---------------------|------------------------|----------------------|-----------------|-------------|-----------------|
| 12 | 64 | 0.0457 | 0.2279 | 0.4627 | 3.56 | 7.230 | ×0.5 |
| 15 | 32 | 0.0480 | 0.8611 | 0.2972 | 26.91 | 9.287 | ×3 |
| 18 | 8 | 0.0598 | 1.8326 | 0.0950 | 229.08 | 11.879 | ×19 |


### Sim caveat

At **n=12** a pure-Python clause→mask loop can be *slower* than NumPy `exp(-iγE)` on
length-4096 (table above). That is a **simulator artifact**. Hardware/GRAPE story:
library is **26 fixed SNAP masks compiled once**; densify allocates/grows as 2^n
(n18: 1 MB/E float32, n=24 would be 64 MB/E). Library per-inst time stays ~O(m L)
with m≈10n; densify phase already ×60 slower by n18 in this CPU bench.

### Memory note

| n | dim=2^n | one E float32 |
|---|---------|---------------|
| 12 | 4096 | 0.02 MB |
| 15 | 32768 | 0.13 MB |
| 18 | 262144 | 1.05 MB |

Offline library: **26 × 8 floats** (~1 KB) + champ γ (32 floats).

## Honest one-liner

Offline SNAP library compiles **Uc = exp(-iγ Hc)** without densifying E;
it does **not** remove Hc — it renames densify as clause→mask schedule
(see `docs/ROBUST_GRAPE.md`).

## Artifacts

- JSON: `COMPILE_OFFLINE.json`
- Library catalog: `LIBRARY.md`
- Two-cav: `COMPILE_TWOCAV.md`
- Freeze: `FREEZE.md`


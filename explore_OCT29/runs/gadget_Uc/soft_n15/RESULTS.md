# Soft-include n15 on projector-Uc L32

_Generated 2026-10-30 20:27:33 +0800 (Asia/Shanghai)_

n15 train_* empty → soft pool = **val** (40). Eval = scale1000 holdout (different seeds).
Gate: n18 Med < 1.955 OR joint n15+n18 improve without n9/12 blow-up (>×1.05).

## scale1000

| arm | n9 | n12 | n15 | n18 | keep |
|-----|----|-----|-----|-----|------|
| t691215 | 1.105/1.745 | 1.240/2.918 | 1.495/5.837 | 1.900/13.741 | True |
| t91215 | 1.100/1.759 | 1.231/2.919 | 1.483/5.847 | 1.884/13.704 | True |

| L32 champ | 1.109/1.828 | 1.258/3.013 | 1.537/5.767 | 1.995/14.117 | ref |

- **t691215**: keep=True kind=n18_holdout — n18 1.900 < 1.955 (zero-shot); joint n15/n18 improve=True/True n9/12 ok=True/True
- **t91215**: keep=True kind=n18_holdout — n18 1.884 < 1.955 (zero-shot); joint n15/n18 improve=True/True n9/12 ok=True/True

## Verdict

**PROMOTE** `t91215` (best n18; chosen over first-keep) → replace `unsat_params.pt`.

## GRAPE library freeze recommendation

With depth ladder stopped (L32) and span/soft-n15 exhausted or settled, **freeze the projector GRAPE library**:

1. **26 local projectors** on 3-bit cavity (k=1,2,3) + two-cavity \(C_a\otimes C_b\) for (1,3)/(2,2).
2. Exact compile: \(U_Z(\gamma)=\mathrm{SNAP}(\theta)\) on Fock support; BinH \(U_X=W^\dagger\mathrm{SNAP} W\).
3. Instance only schedules which library ops fire (clause list) + shared \(\gamma_\ell,\phi_{pc}\).
4. Do **not** GRAPE densified `diag(E)` as a pulse target.
5. Champ angles: L=32 shell SNAP on BinH; see `grape_library/LIBRARY.md`.

Next explore (if any): hardware GRAPE fidelity for two-cavity tensors, not more classical depth.


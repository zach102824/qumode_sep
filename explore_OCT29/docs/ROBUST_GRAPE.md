# ROBUST_GRAPE — targets that are *not* \(e^{-i\gamma H_c}\)

_Asia/Shanghai — explore_OCT29 (Zach t2177: projector ≈ solving Hc; want nonstop alternatives)_

## Honest fact (do not paper over)

For TEB 4-SAT with Fock clause projectors \(C_c\) (mutually commuting):

\[
\prod_c\bigl(I+(e^{-i\gamma}-1)C_c\bigr)=\exp(-i\gamma H_c),\qquad H_c=\sum_c C_c.
\]

So the **projector / frozen-26 library path is Hc-free only in densification**
(never materialize the \(2^n\) diagonal). It is **not** instance-blind physics:
GRAPE still implements the instance cost unitary. Same Med/B90 as Arm C is expected.

## Taxonomy

| Target | Equals \(\exp(-i\gamma H_c)\)? | Instance in pulse? | Status |
|--------|-------------------------------|--------------------|--------|
| Arm C densified \(\mathrm{diag}(E_{\mathrm{unsat}})\) | YES | yes (via \(E\)) | baseline |
| Clause projectors / frozen 26 | YES (exact) | yes (clause→\(C\)) | champ classical; **rename of Hc** |
| XOR / parity gadgets \(\prod\exp(-i\gamma P_c)\) | NO (parity cost) | yes (clauses) | **FAILED** Med≪ArmC |
| Fixed stencil \(\prod_j\exp(-i\gamma X_j)\) | NO | **no** | **FAILED** Med\(\sim 2^{n-1}\) |
| Mixer-only (no cost layer; SNAP shell) | NO | **no** | probe |
| Random fixed diag (shared fake \(E\)) | NO | **no** | probe (control) |
| Shuffle-\(E\) (true \(E\) permutes) | NO | scrambled | probe (ablation) |
| SNAP+D only / beamsplitter library | NO | device-blind | listed; not re-run here |
| Measurement-side cost (post-select / WalkSAT) | N/A | classical | out of GRAPE target |

## Already FAILED (cite numbers, L=16 scale1000)

From `runs/gadget_Uc/results.json`:

| arm | n9 Med/B90 | n12 | n15 |
|-----|------------|-----|-----|
| parity | 310/2159 | 1913/11832 | 13150/87786 |
| stencil_x | 256/512 | 2048/4096 | 16384/32768 |
| Arm C / projector | ~1.4/3.3 | ~2.0/6.3 | ~2.9/11 |

Blind stencil ≈ uniform over computational subspace (Med \(\approx 2^{n-1}\)).

## What “robust for hardware story” means here

A GRAPE **pulse library** is robust iff the compiled targets **cannot reconstruct**
instance Hc even with classical scheduling of which pulses fire.
- **Robust (hardware-story)**: fixed stencil, mixer-only, SNAP+D / BS library, random diag.
- **Not robust (renames Hc)**: projector gadgets, densified Arm C, any product of clause projectors.

If robust arms stay at Med \(\gg\) Arm C, the honest paper claim is:
**classical schedule of frozen clause projectors (= Uc) + shared mixer angles**,
not “GRAPE invents a SAT solver without Hc”.


## Offline compile (Opus #6)

Compile the 26 locals **once** as SNAP / W†SNAP W (Fid=1 exact); γ from champ L32
reuses `θ=γ·mask`. Does **not** make GRAPE Hc-free — only skips densifying E.
See `docs/COMPILE_OFFLINE.md` (mirror: `runs/gadget_Uc/grape_library/COMPILE_OFFLINE.md`).

**One-liner:** offline SNAP library = rename of `Uc=exp(-iγ Hc)`, not a blind cost.

## Probe script

`scripts/probe_robust_grape.py` → `runs/robust_grape/` (verdict table).

## Probe results (2026-10-30 ~24:00 CST, L=16 train{9,12} scale1000)

| arm | =exp(Hc)? | n9 Med/B90 | n12 | n15 |
|-----|-----------|------------|-----|-----|
| Arm C / projector | YES | ~1.4/3.3 | ~1.9–2.0/6 | ~2.8–2.9/11 |
| parity (prior FAIL) | NO | 310/2159 | 1913/11832 | 13150/87786 |
| stencil_x (prior FAIL) | NO | 256/512 | 2048/4096 | 16384/32768 |
| **mixer_only** | NO | 256/512 | 2048/4096 | 16384/32768 |
| **random_diag** | NO | 314/2537 | 2145/14285 | 13973/88453 |
| **shuffle_E** | NO | 1797/13156 | 5864/57137 | 24929/233821 |
| **universal_cover** (26 locals) | NO (=global phase) | 256/512 | 2048/4096 | 16384/32768 |
| **trunc_Hc** (m/2) | NO (partial Hc) | 20.8/65.0 | 63.2/215 | 192/912 |

### Verdict

1. Anything that cannot reconstruct instance Hc stays at Med \(\gg\) projector.
2. All-26 universal cover is analytically \(e^{-i7\gamma}I\) — useless as cost.
3. Truncated Hc helps vs blind but does not recover Arm C.
4. **Promote for hardware:** projector library (= Uc) knowing it **is** Hc. Do not claim Hc-free GRAPE.

Artifacts: `runs/robust_grape/RESULTS.md`.

Tracked copies: `docs/ROBUST_GRAPE_RESULTS.md`, `docs/robust_grape_results.json`.

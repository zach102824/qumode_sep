# N3INIT non-ideal \(W_L\) / SNAP-pad leakage probe

Bounded development. Original shared-Uc Nov3 family only. Frozen v4 center-4. SPSA only. Energy selection. Fidelity post hoc. No GR patch. No candidacy file. Holdout FAIL and later REJECTs still stand. CUTOFF_ROBUST_L1 (ideal pad) still stands and is **not** a hardware claim. Do not re-run ideal-pad copies. Do not patch GR.

This is **not** an optimizer retune. It asks whether a controlled, protocol-justified non-ideal \(W_L\) shrinks or flips the L=1 Nov3−QAOA edge.

## Why not D2 pad-SNAP alone

DECISIONS D2 lists periodic \(\theta_{f\bmod 8}\) as an alternative to SNAP phase 0 on \(f\ge 8\). With ideal \(W_L=H^{\otimes 3}\oplus I\) that channel **cannot** populate the pad (PR106: \(p_{\rm leak}\sim 10^{-15}\)). Using it would repeat the ideal-pad tautology. It is not the probe.

No frozen SNAP–D compile of the full 8×8 BinH \(W\) on \(\mathrm{NF}\ge 10\) exists in-repo. `fit_snap_D_to_target` is cropped \(a\) on 8 levels (not a displacement). `compile_flat_prep` fits \(|0\rangle\to|+\rangle\) only. Inventing a new pulse compile would be a new circuit, not a one-knob probe.

## Channel (one knob)

PROTOCOL §3.1: in binary Fock, the unique \(a^\dagger-a\) edge that leaves the code is **7–8**. Real \(\alpha\) displacement is a real orthogonal matrix. The smallest code-faithful non-ideal \(W_L\) is therefore a **real Givens residual on \(\mathrm{span}\{|7\rangle,|8\rangle\}\)** applied after ideal \(W_L\):

\[
W_\varepsilon = G_{7,8}(\varepsilon)\,W_{\rm BinH},\qquad
G_{7,8}(\varepsilon)=\begin{pmatrix}\cos\varepsilon&-\sin\varepsilon\\ \sin\varepsilon&\cos\varepsilon\end{pmatrix}
\text{ on }\{|7\rangle,|8\rangle\},
\quad \varepsilon=0.10.
\]

\(\varepsilon=0\) recovers PR106. \(\varepsilon=0.10\) is about 1% \(|7\rangle\to|8\rangle\) per \(W\), large enough to see, not a depolarizing or SYK noise model. SNAP stays D2 (phase 0 on \(f\ge 8\)). Uz and \(H^{\rm ext}\) unchanged. NF=10 so the pad exists.

**Circuit order.** Non-ideal \(W^2\neq I\), so the Nov3rd native order is required:

\[
|0\rangle\to\big[W_\varepsilon\,U_{cZ}\,W_\varepsilon\to S\big]^L\to W_\varepsilon.
\]

Ideal arm (A) keeps the PR106 rewrite \(|+\rangle\to[U_{cZ}\to WSW]^L\) with \(\varepsilon=0\), which matches native order at \(\varepsilon=0\).

QAOA (C) uses the **same** \(W_\varepsilon\) embedding: mixer \(W_\varepsilon\,\mathrm{SNAP}(s_1=-2\beta)\,W_\varepsilon\), cost Uz \(J\) only. At the algebraic center \((\gamma,\beta)=(0.25,0.25)\) this coincides with Nov3 start 0.

## Seeds

Exclusive unread band `10528..10535`, n=8. Verified before any instance or SPSA:

- Not local R30 `10504..10519`.
- Not FT cutoff `10520..10527`.
- Not FT V(M) `10480..10487`.
- Not LE `10432..10439`.
- Not FT/prior \(\le 10463\).
- Not holdout `10144..10151`.
- Not validation `11000..11031`.
- Not GR/final `20000..20999`.
- No existing `Nov3_INIT/results/**` file for these seeds.

\(B^*\) is new, classical, interior entropy max on `0..floor(N/2)−1`, not a quantum query.

## Grid

New directory `results/wl/`. N=15, L=1, `Q=1024`. Frozen v4 center-4. NF=10. Fail any cell → sensitive; no GR.

| arm | circuit | method | method_id | \(\varepsilon\) | order |
|---|---|---|---:|---:|---|
| A `nov3_ideal` | shared-shell Nov3, PR106 rewrite | Nov3rd | 1 | 0 | rewrite |
| B `nov3_wl` | shared-shell Nov3, native order | Nov3rd | 9 | 0.10 | hardware |
| C `qaoa_wl` | QAOA p=1, same \(W_\varepsilon\) mixer | QAOA | 0 | 0.10 | hardware |

## Gates (prespecified, energy only, before any SPSA)

Bootstrap 4000, `default_rng(3334)`, seeds resampled.

**Primary:** under non-ideal \(W_\varepsilon\), Nov3−QAOA mean \(\Delta E\) CI entirely `< 0` → **WL_ROBUST_L1**. Else **WL_SENSITIVE**.

**Secondary:** non-ideal Nov3 mean E versus ideal Nov3 mean E (does non-ideal hurt Nov3). Report \(p_{\rm leak}\) on A/B/C.

Any infeasible cell → **WL_SENSITIVE**. Either label: no hardware physical-edge claim; no GR; compiled SNAP–D pulse sequence remains unmodelled beyond this Givens residual.

## Queries

Each cell ledger equals `Q=1024`. Inclusive: 8 seeds × 1024 × 3 arms = 24576 new. Report cumulative total `2793472 + new`. Fit workers 4, BLAS 1.

## Off limits

GR PR #77. Validation. Final. Nov9, SYK, AF, N21, paper \(\mathcal{V}(M)\), ideal-pad re-runs, L=3 SPSA tweaks, extra agents, candidacy notes, frozen-benchmark retune, unrelated noise models.

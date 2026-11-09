# N3INIT Fock-cutoff / leakage validation

Bounded development. Original shared-Uc Nov3 only. Frozen v4 center-4 recipe. SPSA only. Energy selection. Fidelity post hoc. No GR patch. No candidacy file. Holdout FAIL and every later REJECT (pair-shell, shell-gain, spread8, anneal, V(M)) still stand. Do not re-fit `10368..10375`. Do not patch GR. Stop paper-V(M) / pair-phase variants and L=3 SPSA schedule/start-count/freeze-order tweaks.

This is **not** an optimizer retune. It asks whether the shallow Nov3 energy edge is a Fock-truncation artifact.

## Default cutoff D0 (frozen v4)

v4 center-4 sims use the logical rewrite of Nov3rd:

\[
|+\rangle^{\otimes N}\to\big[U_{cZ}\to WSW\big]^L
\]

That rewrite is exact on **Fock 0..7 per cavity** (binary 3-bit code). Hilbert dimension per cavity **D0 = 8**. `Nov_3rd/settings.py` locks `NF = 8`. There is no oscillator pad in those sims: Nov3rd “has no leakage by construction” (PROTOCOL §2.3).

QAOA `p=L` is a qubit circuit (dim \(2^N\)). It has no Fock cutoff. Cells labeled D0 / D0+2 for QAOA are the same qubit recipe, paired for \(\Delta E\) at each Nov3 cutoff.

## Controlled cutoffs

\(D\in\{D0, D0+2\}=\{8,10\}\). **D0+4 is not run** (optional; not required for the primary gate).

Nov3 at \(D>8\) uses the existing FT off-code rules, not a compiled mixer:

- Ideal \(W_L=H^{\otimes 3}\) on levels 0..7, **identity pad** on 8..\(D-1\) (`build_W_BinH`).
- SNAP phase 0 on \(f\ge 8\) (D2).
- \(U_{cZ}\) Uz: a pair phase acts only when both cavities are in the code (D3).
- Energy is per-term \(H^{\rm ext}\) (D4).
- Fidelity unconditioned, no leakage renormalization.

**Compiled \(W_L\) and its leakage are not modelled (D5).** A CUTOFF_ROBUST label is pad-robustness of ideal \(W_L\), not a hardware-feasibility claim, and it does not authorize GR.

Hypothesis: if the L=1 edge is physical in this ideal-\(W_L\) model, raising the pad to \(D0+2\) preserves L=1 Nov3−QAOA mean \(\Delta E\) CI entirely \(<0\).

## Leakage diagnostics (per selected state)

- \(p_{\rm leak}=1-P(\)all cavities in \(0..7)\).
- \(\bar n=\) mean photon number per cavity \(\mathbb{E}[n_c]\).
- \(p_{\rm highest}=1-P(\)all \(n_c < D-1)\).
- \(\lVert\psi\rVert^2\) and norm deficit \(|1-\lVert\psi\rVert^2|\).

QAOA: \(p_{\rm leak}=0\) by construction; \(\bar n\) is mean Hamming weight per 3-qubit cavity.

## Seeds

Exclusive unread band `10520..10527`, n=8. Provenance, verified before any instance or SPSA:

- Not local R30 holdout-style `10504..10519`.
- Not FT V(M) `10480..10487`.
- Not LE N21 `10432..10439`.
- Not FT/prior `≤10463`.
- Not holdout `10144..10151`.
- Not validation `11000..11031`.
- Not GR/final `20000..20999`.
- No existing `Nov3_INIT/results/**` file for these seeds.

The runner refuses every other seed. \(B^*\) is new, classical, interior entropy max on `0..floor(N/2)−1`, not a quantum query.

## Grid

New directory `results/cutoff/`. N=15. Frozen v4 center-4, four equal SPSA blocks, Spall \(c=0.1\), \(A=25\), \(\alpha=0.602\), \(\gamma=0.101\). Energy-only selection. Fail any cell → sensitive/fail; no GR.

| arm | circuit | method | nf | L=1 Q | L=2 Q |
|---|---|---|---:|---:|---:|
| `nov3_D0` | shared-shell Nov3 Fock | Nov3rd | 8 | 1024 | 2048 |
| `nov3_D2` | shared-shell Nov3 Fock | Nov3rd | 10 | 1024 | 2048 |
| `qaoa_D0` | QAOA \(p=L\) qubit | QAOA | 8 (label) | 1024 | 2048 |
| `qaoa_D2` | QAOA \(p=L\) qubit | QAOA | 10 (label) | 1024 | 2048 |

L=1 is the primary edge replication. L=2 is sanity / leakage trend only. Same \(Q=1024\cdot L\) on every arm at that L. method_id Nov3=1, QAOA=0 (v4). SPSA RNG does **not** include `nf`, so a Nov3 landscape that is independent of the pad yields the same trajectory at D0 and D0+2.

## Gates (prespecified, energy only, before any SPSA)

Bootstrap 4000, `default_rng(3334)`, seeds resampled.

**Primary:** at L=1, Nov3−QAOA mean \(\Delta E\) CI entirely `< 0` at **both** D0 and D0+2 → **CUTOFF_ROBUST_L1**. Else **CUTOFF_SENSITIVE**.

**Secondary:** report \(|\mathrm{mean}\,\Delta E(D0+2)-\mathrm{mean}\,\Delta E(D0)|\) at L=1. L=2 sign concordance of mean \(\Delta E\) across cutoffs is exploratory and does not decide the label.

Any infeasible cell → **CUTOFF_SENSITIVE**. Either label: do not claim a hardware physical edge; no GR; compiled \(W_L\) remains unmodelled.

## Queries

Each cell ledger equals its `Q`. Inclusive: 8 seeds × (1024+2048) × 4 arms = 98304 new. Report cumulative total `2695168 + new`. Fit workers 4, BLAS 1.

## Off limits

GR PR #77. Validation. Final. Nov9, SYK, AF, N21, paper \(\mathcal{V}(M)\), pair-phase, L=3 SPSA tweaks, re-fit of `10368..10375`, extra agents, candidacy notes, frozen-benchmark retune, compiled-\(W_L\) invention.

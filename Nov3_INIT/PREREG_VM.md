# N3INIT paper \(\mathcal{V}(M)\) architecture arm (arxiv 2204.02923)

Bounded development. **Separate architecture**, not a Nov3 retune. SPSA only. Energy selection. Fidelity post hoc. No GR patch. No candidacy file. Holdout FAIL, pair-shell REJECT, shell-gain REJECT (shared-with-LE-PR101), spread8 REJECT, and anneal REJECT still stand. Do not re-fit `10368..10375`. Do not patch GR. Stop more SPSA gain-schedule / freeze-order / shell-unshare / start-count L=3 tweaks.

This is **not** pair-shell, **not** shared-Uc+SNAP, **not** a hybrid of Nov3 shells with free \(M_{nm}\).

## Paper circuit (implemented)

Source: arXiv 2204.02923 eqs. (3)–(4). The paper’s pair-phase layer is

\[
\mathcal{V}(M)=\exp\Big(-\frac{i}{4}\sum_{n<m}M_{nm}Z_n Z_m\Big).
\]

The paper’s GCS sandwich is \(U(y)\,\mathcal{V}(M)\,|\phi(x)\rangle\) (two free product operators around one \(\mathcal{V}\)). For equal-layer \(p=L\) comparison against QAOA and Nov3, the implemented arm is the paper’s QAOA-resembling core with **independent all-to-all \(M_{nm}\)**:

\[
|+\rangle^{\otimes N}\;\to\;\prod_{\ell=1}^{L}\Big[\mathcal{V}(M^{(\ell)})\;R_x(\beta_\ell)^{\otimes N}\Big].
\]

Mixer is **shared \(\beta_\ell\)** (allowed: “shared or paper-default”). Paper-default full SU(2) product \(U(y),|\phi(x)\rangle\) (6N Bloch angles per sandwich) is **not** implemented; that would not be a \(p=L\) layer count. Smallest paper-faithful subset that still has the independent ZZ layer is this core. No Nov3 \(\gamma,\theta_K,\theta_M\), no SNAP shells, no \(W\).

Layout per layer: upper-triangle \(M\) in row-major \((0,1),(0,2),\dots,(N-2,N-1)\), then \(\beta\). \(N=15\) pairs \(N(N-1)/2=105\). Width \(106\).

Algebraic QAOA reduction: \(M=J\) and \(\beta=0.25\) makes \(\mathcal{V}(J)=\exp(-i\cdot 0.25\sum_{i<j}J_{ij}Z_iZ_j)\), identical to QAOA \(\gamma=0.25\). Start 0 uses this on every layer.

## Differences vs Nov3 shared-Uc+shells

| | (A) Nov3 center-4 v4 | (B) paper \(\mathcal{V}(M)\) | (C) QAOA \(p=L\) |
|---|---|---|---|
| circuit | \(\lvert+\rangle\to[U_{cZ}\to WSW]^L\) | \(\lvert+\rangle\to[\mathcal{V}(M_\ell)\,R_x(\beta_\ell)]^L\) | \(\lvert+\rangle\to[e^{-i\gamma_\ell H_{ZZ}}R_x(\beta_\ell)]^L\) |
| cost params | shared \(\gamma,\theta_K,\theta_M\) | independent \(M_{nm}\) per pair per layer | \(\gamma_\ell\) times \(J\) |
| mixer | 3-shell SNAP (shared) | shared \(\beta_\ell\) | shared \(\beta_\ell\) |
| N=15 L=2 params | 9 | 212 | 4 |
| N=15 L=3 params | 12 | 318 | 6 |
| start 0 | \(\gamma=0.25\), \(s_1=-0.5\) | \(M=J\), \(\beta=0.25\) | \(\gamma=\beta=0.25\) |
| method_id | 1 | 8 | 0 |

Same SPSA (Spall \(c=0.1\), \(A=25\), \(\alpha=0.602\), \(\gamma=0.101\), 4 cal pairs, clip \(2\sigma\), Rademacher), same center-4, four equal blocks, same \(Q=1024\cdot L\). \(\sigma_M=0.15\), \(\sigma_\beta=0.10\). Starts 1–3: box \(U(-0.5,0.5)\) on every coordinate (fair equal4 counterpart, not paper GOE \(M\) draws).

## Seeds

Exclusive unread band `10480..10487`, n=8. Provenance, verified before any instance or SPSA:

- Not local R28 `10464..10479`.
- Not FT/prior `≤10463` (covers anneal `10456..10463`, spread8 `10376..10383`, shell-gain `10368..10375`, pair-shell, LE `10432..10439`, R26/R27, train).
- Not holdout `10144..10151`.
- Not validation `11000..11031`.
- Not GR/final `20000..20999`.
- No existing `Nov3_INIT/results/**` file for these seeds.

The runner refuses every other seed. \(B^*\) is new, classical, interior entropy max on `0..floor(N/2)−1`, not a quantum query.

## Grid

New directory `results/vm/`. N=15, \(L\in\{2,3\}\), `q=1024`, `Q=1024·L`. Three arms. Energy-only selection. Fail any cell → reject; no GR.

| arm | circuit | method | method_id |
|---|---|---|---:|
| A `nov3_center4` | shared-shell Nov3 v4 | Nov3rd | 1 |
| B `paper_vm` | paper \(\mathcal{V}(M)\) \(p=L\) | VM | 8 |
| C `qaoa_center4` | QAOA \(p=L\) | QAOA | 0 |

## Gates (prespecified, energy only, before any SPSA)

Bootstrap 4000, `default_rng(3334)`, seeds resampled.

**Primary L=3:** \(\mathcal{V}(M)\) vs QAOA mean \(\Delta E\) CI entirely `< 0`.

**Secondary L=3:** \(\mathcal{V}(M)\) mean E `≤` Nov3 mean E (not worse than Nov3).

**L=2 sanity / regression only:** \(\mathcal{V}(M)\) mean E `≤` Nov3 mean E, and \(\mathcal{V}(M)\) vs QAOA CI is **not** entirely `> 0`.

Fail primary, secondary, L=2 sanity, or any infeasible cell → **REJECT_VM_L3**. Pass all → **VM_L3_EDGE**. Either label: no GR patch, no candidacy file.

## Queries

Each cell ledger equals `Q=1024·L`. Inclusive: 8 seeds × (2048+3072) × 3 arms = 122880 new. Report cumulative total `2572288 + new`. Fit workers 4, BLAS 1.

## Off limits

GR PR #77. Validation. Final. Nov9, SYK, AF, N21, Nov3 retune, freeze-order, shell-unshare, gain-schedule, start-count L=3 tweaks, re-fit of `10368..10375`, extra agents, candidacy notes, frozen-benchmark retune, hybrid Nov3+V(M).

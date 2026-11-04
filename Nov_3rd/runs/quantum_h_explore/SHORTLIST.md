# SHORTLIST — Hamiltonian families vs locked constraints

_Asia/Singapore 2026-11-04 ~21:50_

Locks: **G**ate · **Q**uantum GS · **R**eadout Z+X · **U**c^X/GRAPE · **T**ransfer · **B**eat QAOA · **C**ross-cavity 3-local.

Legend: ✅ pass · ⚠️ soft / conditional · ❌ fail (kill).

---

## Ranked families (≥6)

### A — Transverse Max-E (TFIM on clause graphs) — **TOP PICK**
\[
H = H_c(Z) + \lambda \sum_{i=1}^{n} X_i,\qquad
H_c = \sum_{c\in\mathrm{clauses}} \Pi_c^{\mathrm{unsat}}
\]
(reuse existing unsat \(E\) as diagonal of \(H_c\); optional affine map \(H_c\leftarrow E\)).

| G | Q | R | U | T | B | C |
|---|---|---|---|---|---|---|
| ✅ | ✅ (λ>0) | ✅ | ✅ | ✅ | ✅ (test) | ✅ |

- **Why method should win:** \(U_c^X\) is *exactly* the cost layer of current motif; Hamming-shell SNAP ≈ mixer; Walsh-degree / power_p already beat classical QAOA on same \(H_c\) — transverse field is a controlled deformation, not a rewrite.
- **λ guess:** smoke `{0, 0.25, 0.5, 1.0, 2.0}` × mean\|clause weight\|; pick smallest λ with bitstring overlap of GS < 0.5 and classical greedy/Med gap opens.
- **Kill:** at all λ>0, SNAP never ≤0.98× QAOA on n6&n9; or GS stays classical (λ too small).

### B — Sparse quantum \(p=3\) spin TFIM
\[
H = -\sum_{(i,j,k)\in\mathcal{T}} J_{ijk}\, Z_i Z_j Z_k \;-\; \lambda\sum_i X_i
\]
\(\mathcal{T}\) = sparse 3-uniform hypergraph (can = clause supports; \(J=\pm 1\) or Mag-glass).

| G | Q | R | U | T | B | C |
|---|---|---|---|---|---|---|
| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

- **Why win:** native 3-local matches cavity packing; literature “quantum p-spin”; no Onsager formula.
- **λ guess:** `{0.3, 0.7, 1.0, 1.5}` with \(J=\pm 1\) normalized by \(\sqrt{|\mathcal{T}|/n}\).
- **Kill:** regenerating \(\mathcal{T}\) breaks pack reuse and transfer vs A without beating A’s R.

### C — Power-law long-range TFIM (1D chain embedding)
\[
H = -\sum_{i<j}\frac{J}{|i-j|^\alpha} Z_i Z_j \;-\; \lambda\sum_i X_i
\]
generic \(\alpha\) (smoke \(\alpha\in\{1.5, 2.0, 3.0\}\)); **no** closed-form \(e(\lambda,\alpha)\).

| G | Q | R | U | T | B | C |
|---|---|---|---|---|---|---|
| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ (2-local; cross-cavity only via embedding) |

- **Why win:** clean lit baseline; still Z+X readout; \(U_c^X\) from full ZZ diag.
- **λ guess:** critical-ish \(\lambda\sim O(1)\) for α≳2; grid `{0.5, 1.0, 1.5, 2.0}`.
- **Kill:** loses cross-cavity 3-local story / Max-E narrative; only keep if A/B fail.

### D — Signed random 3-local Ising + transverse (**runner**)
\[
H = \sum_{c=1}^{m} J_c\, Z_{i_c} Z_{j_c} Z_{k_c}\;(+Z_{\ell_c}) \;+\; \lambda\sum_i X_i,\quad J_c\sim\mathrm{Unif}\{\pm 1\}
\]
supports from existing `cls_v` (drop 4th lit or keep 4-local subset); signs from `cls_s` or re-draw.

| G | Q | R | U | T | B | C |
|---|---|---|---|---|---|---|
| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

- **Why win:** frustrates classical GS harder than unsigned unsat; same pack geometry; GRAPE of signed \(H_c\) still one diagonal pulse.
- **λ guess:** `{0.25, 0.5, 1.0, 2.0}`; compare unsigned A at same λ.
- **Kill:** if signing does not improve SNAP−QAOA margin vs A, prefer A (simpler story).

### E — Stoquastic vs non-stoquastic variants
- **E1 stoquastic (keep):** \(H = H_c(Z) - \lambda\sum X_i\) with \(H_c\) real diagonal, λ>0 — standard TFIM-like; QMC-friendly; **passes R**.
- **E2 non-stoquastic (usually kill under R):** add \(+\mu\sum Y_i\), or \(\nu\sum_{\langle ij\rangle} X_i X_j\), or complex phases on off-diag.
  - \(Y\)-field → need Y-hist (❌ R as locked).
  - \(XX\) edges → need \(\langle X_i X_j\rangle\) (❌ R).

| variant | G | Q | R | U | T | B | C |
|---------|---|---|---|---|---|---|---|
| E1 −λΣX | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| E2 +Y / +XX | ✅ | ✅ | ❌ | ✅ | ✅ | — | ⚠️ |

- **Use:** prefer **E1 sign convention** inside A/B/D (field = \(-\lambda\sum X\) lowering energy). E2 only if Zach later expands readout.
- **Kill E2** under current Z+X lock.

### F — Quantum MaxCut / Heisenberg (**KILL**)
\[
H = \sum_{\langle ij\rangle}\bigl(I - X_i X_j - Y_i Y_j - Z_i Z_j\bigr)
\]

| G | Q | R | U | T | B | C |
|---|---|---|---|---|---|---|
| ⚠️ | ✅ | ❌ | ⚠️ | ✅ | — | ⚠️ |

- Needs \(XX+YY\) correlators → **fails R**.
- \(U_c^X\) from ZZ piece alone ≠ full cost layer → weakens **U** story.
- **Kill** for this explore (keep as negative control one-liner in smoke docs only if useful).

### G — Two noncommuting cost pieces (alt, not top-3)
\[
H = H_A(Z) + H_B(X),\quad [H_A,H_B]\neq 0
\]
e.g. clause \(H_A\) + ring-\(XX\) \(H_B\). Providing only \(U_A^X\) is honest, but \(\langle H_B\rangle\) needs XX correlators → **fails R** unless \(H_B=\lambda\sum X_i\) (then = A).

| G | Q | R | U | T | B | C |
|---|---|---|---|---|---|---|
| ✅ | ✅ | ❌/✅ | ✅ | ✅ | — | ✅ |

- Collapses to **A** when \(H_B\) is a field. Otherwise kill under R.

---

## TOP 3 for smoke

| rank | family | exact \(H\) | λ / knobs | win vs QAOA | kill |
|------|--------|-------------|-----------|-------------|------|
| **1** | **A** transverse Max-E | \(H_c^{\mathrm{unsat}}(Z)+\lambda\sum X_i\) | λ∈{0,0.25,0.5,1,2} | reuse motif+Walsh/power_p; QAOA on same \(H\) | no BEAT any λ>0; or GS classical |
| **2** | **D** signed 3-local + X | \(\sum J_c Z_i Z_j Z_k +\lambda\sum X_i\) | λ∈{0.25,0.5,1,2}, \(J=\pm1\) | frustration ↑; same packs | margin ≤ A |
| **3** | **B** sparse p=3 TFIM | \(-\sum_{\mathcal{T}} J ZZZ -\lambda\sum X\) | λ∈{0.3,0.7,1,1.5}, \|\(\mathcal{T}\)\|~m | lit name; 3-local native | worse reuse / no gain vs A |

**Reserve:** C power-law if A/D/B all fail transfer narrative.
**Explicit kill:** F Quantum MaxCut; E2 non-stoquastic needing Y/XX.

## Metric (all TOP 3)
- Primary smoke: median \(\langle H\rangle\) and \(\langle H\rangle/n\) on B_hold; gap \(\langle H\rangle-E_0\) at n6/n9 (exact diag).
- BEAT: Med ≤ 0.98 × Fourier QAOA same L; report R = Med_SNAP / Med_QAOA.
- Sanity λ=0 on A: should recover classical Max-E behavior (bitstring GS).

## Recommendation
**Start smoke on A** (least new code: \(E\) packs + λX). If SNAP≻QAOA for some λ>0 with entangled GS → **lock A** and only then try D (sign) as ablation. B if we want the “quantum p-spin” paper framing. Skip F.

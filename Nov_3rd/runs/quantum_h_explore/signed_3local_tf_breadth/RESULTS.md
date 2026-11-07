# Signed 3-local transverse field — breadth smoke

_Verdict: **KILL**._

- L=2 median-seed R_Fourier: λ=0.50 L=2 R(n6,n9,n12)=(1.028,1.013,1.013); λ=0.75 L=2 R(n6,n9,n12)=(1.027,1.011,1.012); λ=1.00 L=2 R(n6,n9,n12)=(1.013,1.009,1.010)
- all λ fail quantumness (need median cavity Λ≤0.80 and product gap/n≥0.03 at n=6, 9, and 12)
- L=2 R_Fourier also exceeds 0.98 at both n6 and n9 for every λ, so the motif kill fires as well. needs_edit does not apply: those ratios sit above the soft band. L=4 shell polish can beat Fourier on the train sizes, but the pass gate is L=2 at n12 and no scored λ is quantum.

## Hamiltonian

Family D from `SHORTLIST.md`, μ = 0 (no cavity XX):

\[
H = \sum_c J_c Z_{i_c} Z_{j_c} Z_{k_c} \;-\; \lambda \sum_{i=1}^{n} X_i,
\qquad J_c\in\{+1,-1\}.
\]

Supports come from the sealed 4-SAT packs `explore_OCT29/data/data_n{6,9,12}.npz`. Each clause keeps its three lowest-index literals; the fourth literal is dropped so every term is strictly 3-local. `J_c` is the product of those three `cls_s` signs. There is no μ Σ XX inside a cavity — that term can pin the ground state to a product of cavity states and push cavity Λ back to 1.

Cross-cavity fraction of the kept triples, averaged over 16 val instances: n6=0.782, n9=0.933, n12=0.964.

`H_Z` is a pure Walsh degree-3 function, so graded γ weights on degrees 1, 2, and 4 are identically idle. Walsh SNAP trains γ_3 per layer plus the 4 Hamming-shell phases. Fourier QAOA uses Zhou modes of (β, γ) with the locked mixer ⊗ exp(−i β X/2) and cost phase exp(−i γ H_Z n/m). L=2 with q=2 is full-rank (same family as free angles). L=4 uses q=3.

## Protocol

- λ ∈ {0.50, 0.75, 1.00}, L ∈ {2, 4}, seeds {0, 1, 2}.
- Train on train[:32] of n=6 and n=9 jointly (one angle set). Freeze and score val of n=6, n=9, and n=12. n12 is never trained. n21 is not loaded.
- Walsh is warm-started from that seed's trained Fourier angles (γ_3 ← γ, φ_j ← −β j), then L-BFGS-polished on the same ⟨H⟩/n objective. That shell is the QAOA mixer: H exp(i β |1⟩⟨1|) H = e^{iβ/2} exp(−i β X/2), so φ_j=−β j reproduces ⊗ exp(−i β X/2) up to a global phase. Polish can bend the four shell phases off that ray.
- Optimizer: L-BFGS-B only.

### Metrics

- **Cavity Λ** (true GS): largest eigenvalue of the 8-level reduced state of each cavity (3 qubits). Headline number is the median over val instances and cavities. A cavity-product state has Λ = 1. Quantum gate: median Λ ≤ 0.80 at every n ∈ {6, 9, 12}.
- **Product gap**: (E_MF − E_0) / n, where E_MF is the best qubit-product energy from multi-start coordinate descent (bitstring vertices included). Gate: median gap ≥ 0.03 at every n. The classical bitstring gap (min_x H_Z(x) − E_0)/n is listed separately and is not the gate: a tilted product state lies below the best bitstring while remaining separable.
- **Z-peak**: max_x |⟨x|ψ_GS⟩|². Reported with the quantumness numbers; not an extra gate.
- **R_Fourier**: median_instances(⟨H⟩−E_0)_Walsh / median_instances(⟨H⟩−E_0)_Fourier, then the median of that ratio over the 3 paired seeds. Lower means Walsh is closer to the true ground state. Pass needs L=2 R ≤ 0.85 at n12 on a λ that is quantum. R in (0.85, 0.98] is soft. R > 0.98 at both n6 and n9 for every λ kills the motif.

## Ground state

| λ | n | median Λ | p10 Λ | median gap/n | median classical gap/n | median Z-peak | quantum |
|---|---|----------|-------|--------------|------------------------|---------------|---------|
| 0.50 | 6 | 1.000 | 0.729 | 0.0001 | 0.0986 | 0.496 | no |
| 0.50 | 9 | 1.000 | 0.999 | 0.0000 | 0.0715 | 0.493 | no |
| 0.50 | 12 | 1.000 | 1.000 | 0.0001 | 0.0613 | 0.484 | no |
| 0.75 | 6 | 1.000 | 0.778 | 0.0003 | 0.1597 | 0.490 | no |
| 0.75 | 9 | 1.000 | 0.999 | 0.0002 | 0.1191 | 0.484 | no |
| 0.75 | 12 | 1.000 | 0.999 | 0.0003 | 0.1061 | 0.465 | no |
| 1.00 | 6 | 0.999 | 0.810 | 0.0011 | 0.2289 | 0.480 | no |
| 1.00 | 9 | 1.000 | 0.996 | 0.0006 | 0.1746 | 0.472 | no |
| 1.00 | 12 | 1.000 | 0.997 | 0.0009 | 0.1587 | 0.440 | no |

Root-mean-square of H_Z on the hypercube (λ=0.50 block, same diagonal for every λ): n6=7.07, n9=9.06, n12=11.00. Requested λ ∈ {0.50, 0.75, 1.00} sits well below that scale.

## Walsh vs Fourier

Residuals are median_(val) (⟨H⟩ − E_0) / n. R is the median over seeds of (Walsh median / Fourier median).

| λ | L | R n6 | R n9 | R n12 | Walsh resid n12 | Fourier resid n12 |
|---|---|------|------|-------|-----------------|-------------------|
| 0.50 | 2 | 1.028 | 1.013 | 1.013 | 1.3485 | 1.3312 |
| 0.50 | 4 | 0.850 | 0.920 | 0.993 | 1.2149 | 1.2240 |
| 0.75 | 2 | 1.027 | 1.011 | 1.012 | 1.2494 | 1.2345 |
| 0.75 | 4 | 0.830 | 0.884 | 0.925 | 0.9656 | 1.0434 |
| 1.00 | 2 | 1.013 | 1.009 | 1.010 | 1.1479 | 1.1368 |
| 1.00 | 4 | 0.808 | 0.880 | 0.915 | 0.8873 | 0.9695 |

### Per seed, L=2 (paired)

| λ | seed | R n6 | R n9 | R n12 | train loss Fourier init→final | Walsh init→final |
|---|------|------|------|-------|--------------------------------|------------------|
| 0.50 | 0 | 1.028 | 1.013 | 1.013 | -1.2846→-1.6375 | -1.6375→-1.6580 |
| 0.50 | 1 | 1.028 | 1.013 | 1.013 | -1.2220→-1.6375 | -1.6375→-1.6580 |
| 0.50 | 2 | 1.028 | 1.013 | 1.013 | -1.1473→-1.6375 | -1.6375→-1.6580 |
| 0.75 | 0 | 1.027 | 1.011 | 1.012 | -1.3953→-1.7907 | -1.7907→-1.8087 |
| 0.75 | 1 | 1.027 | 1.011 | 1.012 | -1.2661→-1.7907 | -1.7907→-1.8087 |
| 0.75 | 2 | 1.027 | 1.011 | 1.012 | -1.4542→-1.7907 | -1.7907→-1.8087 |
| 1.00 | 0 | 1.013 | 1.009 | 1.010 | -1.5879→-1.9529 | -1.9529→-1.9687 |
| 1.00 | 1 | 1.013 | 1.009 | 1.010 | -1.6859→-1.9529 | -1.9529→-1.9687 |
| 1.00 | 2 | 1.013 | 1.009 | 1.010 | -1.5210→-1.9529 | -1.9529→-1.9687 |

## Diagnostic λ scan (not scored)

Same Hamiltonian on the first 8 val instances. This grid is not part of the pass rule; it only shows where cavity Λ and the product gap move.

| λ | n | median Λ | median gap/n | median Z-peak |
|---|---|----------|--------------|---------------|
| 0.25 | 6 | 1.000 | 0.0000 | 0.493 |
| 0.50 | 6 | 0.998 | 0.0003 | 0.474 |
| 0.75 | 6 | 0.993 | 0.0013 | 0.446 |
| 1.00 | 6 | 0.980 | 0.0041 | 0.415 |
| 1.50 | 6 | 0.900 | 0.0265 | 0.334 |
| 2.00 | 6 | 0.832 | 0.1059 | 0.170 |
| 3.00 | 6 | 0.919 | 0.2387 | 0.064 |
| 4.00 | 6 | 0.958 | 0.2147 | 0.042 |
| 6.00 | 6 | 0.982 | 0.1407 | 0.029 |
| 0.25 | 9 | 1.000 | 0.0000 | 0.495 |
| 0.50 | 9 | 1.000 | 0.0000 | 0.482 |
| 0.75 | 9 | 1.000 | 0.0001 | 0.465 |
| 1.00 | 9 | 1.000 | 0.0004 | 0.446 |
| 1.50 | 9 | 0.999 | 0.0017 | 0.406 |
| 2.00 | 9 | 0.997 | 0.0055 | 0.366 |
| 3.00 | 9 | 0.945 | 0.0420 | 0.247 |
| 4.00 | 9 | 0.873 | 0.2366 | 0.056 |
| 6.00 | 9 | 0.949 | 0.2666 | 0.012 |

## Decision rule

PASS if some scored λ has median cavity Λ ≤ 0.80 and product gap/n ≥ 0.03 at n=6, 9, and 12, and L=2 median-seed R_Fourier ≤ 0.85 at n12. needs_edit if quantumness holds but that R is soft. KILL if every scored λ fails quantumness, or if every λ has L=2 R > 0.98 at both n6 and n9.

**Verdict: KILL.**

## Scope

Wall time 80s. Train B=32, L-BFGS maxiter=60. Did not modify `Nov_3rd/settings.py`, did not write `qaoa_angles/`, and did not run or signal n21.

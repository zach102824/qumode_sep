# Sparse p=3 TFIM — breadth smoke

_Verdict: **KILL**._

- L=2 median-seed R_Fourier: λ=0.50 L=2 R(n6,n9,n12)=(0.997,0.999,0.999); λ=1.00 L=2 R(n6,n9,n12)=(0.997,0.999,0.999); λ=1.50 L=2 R(n6,n9,n12)=(0.978,0.999,1.021); λ=2.00 L=2 R(n6,n9,n12)=(0.971,1.007,1.029)
- all λ fail quantumness (need median cavity Λ≤0.80 and product gap/n≥0.03 at n=6, 9, and 12)

## Hamiltonian

Family B from `SHORTLIST.md`, μ = 0 (no cavity XX). NEW_SAFE (unsat transverse field, λ = 0.75, μ = 0) stays the locked claim; this file is a breadth smoke only.

\[
H = -\sum_{(i,j,k)\in T}\frac{J_{ijk}}{s}\, Z_i Z_j Z_k \;-\; \lambda \sum_{i=1}^{n} X_i,
\qquad s=\sqrt{|T|/n},\quad J_{ijk}\in\{+1,-1\}.
\]

T is a sparse 3-uniform hypergraph reused from the sealed 4-SAT packs `explore_OCT29/data/data_n{6,9,12}.npz`. Each clause offers its four 3-subsets. The kept edge is a still-unused maximum cavity-span triple with the smallest current degree sum (lexicographic tie-break). J is the product of those three literal signs. Repeated triples are not stacked, so every coupling stays ±1 before the 1/s factor. Intra-cavity triples are not used. There is no μ Σ XX.

Median |T| on the scored val split (B=40) is n6=18 (s=1.732), n9=57 (s=2.517), n12=101 (s=2.901). A 16-instance preview of the same builder gives n6: cross=1.000, deg_min=9, n9: cross=1.000, deg_min=16, n12: cross=1.000, deg_min=22. Clause count m is 60 / 89 / 119, so |T|/m is 0.30 / 0.64 / 0.85. At n=6 the kept set is the complete cross-cavity hypergraph (18 of 18). At n=12 it is 101 of C(12,3)=220 triples. Every scored instance has minimum degree at least 1. Signed 3-local (#54) used these packs with a different extraction (drop the highest literal, multiplicity kept) and no 1/s factor, so its λ ~ 1 sat under an unnormalized diagonal of rms 7–11.

`H_Z` is a pure Walsh degree-3 function, so graded γ weights on degrees 1, 2, and 4 are identically idle. Walsh SNAP trains γ_3 per layer plus the 4 Hamming-shell phases. Fourier QAOA uses Zhou modes of (β, γ) with the locked mixer ⊗ exp(−i β X/2) and cost phase exp(−i γ H_Z n/m), m = pack clause count = round(9.93 n). L=2 with q=2 is full-rank (same family as free angles). L=4 uses q=3.

## Protocol

- λ ∈ {0.50, 1.00, 1.50, 2.00}, L ∈ {2, 4}, seeds {0, 1, 2}.
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
| 0.50 | 6 | 0.997 | 0.659 | 0.0013 | 0.0516 | 0.858 | no |
| 0.50 | 9 | 0.997 | 0.967 | 0.0016 | 0.0570 | 0.848 | no |
| 0.50 | 12 | 0.998 | 0.988 | 0.0016 | 0.0571 | 0.779 | no |
| 1.00 | 6 | 0.829 | 0.768 | 0.0451 | 0.2568 | 0.413 | no |
| 1.00 | 9 | 0.823 | 0.723 | 0.0334 | 0.2499 | 0.292 | no |
| 1.00 | 12 | 0.927 | 0.690 | 0.0224 | 0.2308 | 0.308 | no |
| 1.50 | 6 | 0.911 | 0.887 | 0.1180 | 0.6586 | 0.092 | no |
| 1.50 | 9 | 0.888 | 0.861 | 0.1199 | 0.6441 | 0.030 | no |
| 1.50 | 12 | 0.871 | 0.842 | 0.1193 | 0.6102 | 0.013 | no |
| 2.00 | 6 | 0.954 | 0.947 | 0.0864 | 1.1244 | 0.052 | no |
| 2.00 | 9 | 0.942 | 0.934 | 0.0875 | 1.1080 | 0.012 | no |
| 2.00 | 12 | 0.936 | 0.930 | 0.0869 | 1.0752 | 0.003 | no |

Root-mean-square of the normalized H_Z on the hypercube (same diagonal for every λ): n6=2.45, n9=3.00, n12=3.46. The same diagonal before dividing by s would be n6=4.24, n9=7.55, n12=10.05. λ ∈ {0.50, 1.00, 1.50, 2.00} is O(1) against the normalized per-site scale.

## Walsh vs Fourier

Residuals are median_(val) (⟨H⟩ − E_0) / n. R is the median over seeds of (Walsh median / Fourier median).

| λ | L | R n6 | R n9 | R n12 | Walsh resid n12 | Fourier resid n12 |
|---|---|------|------|-------|-----------------|-------------------|
| 0.50 | 2 | 0.997 | 0.999 | 0.999 | 0.3137 | 0.3141 |
| 0.50 | 4 | 0.932 | 0.959 | 0.962 | 0.2509 | 0.2708 |
| 1.00 | 2 | 0.997 | 0.999 | 0.999 | 0.0856 | 0.0857 |
| 1.00 | 4 | 0.690 | 0.855 | 0.933 | 0.0695 | 0.0745 |
| 1.50 | 2 | 0.978 | 0.999 | 1.021 | 0.0097 | 0.0095 |
| 1.50 | 4 | 0.711 | 0.675 | 0.629 | 0.0039 | 0.0063 |
| 2.00 | 2 | 0.971 | 1.007 | 1.029 | 0.0032 | 0.0031 |
| 2.00 | 4 | 0.741 | 0.554 | 0.498 | 0.0008 | 0.0017 |

### Per seed, L=2 (paired)

| λ | seed | R n6 | R n9 | R n12 | train loss Fourier init→final | Walsh init→final |
|---|------|------|------|-------|--------------------------------|------------------|
| 0.50 | 0 | 0.997 | 0.999 | 0.999 | -0.5946→-0.7605 | -0.7605→-0.7610 |
| 0.50 | 1 | 0.997 | 0.999 | 0.999 | -0.5829→-0.7605 | -0.7605→-0.7610 |
| 0.50 | 2 | 0.996 | 1.002 | 1.003 | -0.5677→-0.7709 | -0.7709→-0.7712 |
| 1.00 | 0 | 0.997 | 0.999 | 0.999 | -1.0627→-1.1596 | -1.1596→-1.1598 |
| 1.00 | 1 | 0.993 | 1.000 | 1.007 | -1.0754→-1.1623 | -1.1623→-1.1623 |
| 1.00 | 2 | 0.997 | 0.999 | 0.999 | -1.0663→-1.1596 | -1.1596→-1.1598 |
| 1.50 | 0 | 0.978 | 0.999 | 1.021 | -1.5684→-1.6111 | -1.6111→-1.6111 |
| 1.50 | 1 | 0.978 | 0.999 | 1.021 | -1.5595→-1.6111 | -1.6111→-1.6111 |
| 1.50 | 2 | 0.994 | 0.999 | 0.999 | -1.5783→-1.6104 | -1.6104→-1.6104 |
| 2.00 | 0 | 0.971 | 1.007 | 1.029 | -2.0636→-2.0836 | -2.0836→-2.0837 |
| 2.00 | 1 | 0.971 | 1.007 | 1.029 | -2.0467→-2.0836 | -2.0836→-2.0837 |
| 2.00 | 2 | 0.971 | 1.007 | 1.029 | -2.0621→-2.0836 | -2.0836→-2.0837 |

## Diagnostic λ scan (not scored)

Same Hamiltonian on the first 8 val instances. This grid is not part of the pass rule; it only shows where cavity Λ and the product gap move.

| λ | n | median Λ | median gap/n | median Z-peak |
|---|---|----------|--------------|---------------|
| 0.25 | 6 | 0.998 | 0.0005 | 0.929 |
| 0.50 | 6 | 0.959 | 0.0055 | 0.750 |
| 1.00 | 6 | 0.829 | 0.0789 | 0.273 |
| 1.50 | 6 | 0.919 | 0.1156 | 0.073 |
| 2.00 | 6 | 0.956 | 0.0852 | 0.044 |
| 3.00 | 6 | 0.981 | 0.0561 | 0.030 |
| 4.00 | 6 | 0.989 | 0.0419 | 0.025 |
| 0.25 | 9 | 1.000 | 0.0001 | 0.956 |
| 0.50 | 9 | 0.998 | 0.0015 | 0.840 |
| 1.00 | 9 | 0.828 | 0.0273 | 0.311 |
| 1.50 | 9 | 0.880 | 0.1236 | 0.031 |
| 2.00 | 9 | 0.940 | 0.0880 | 0.012 |
| 3.00 | 9 | 0.976 | 0.0568 | 0.006 |
| 4.00 | 9 | 0.987 | 0.0422 | 0.004 |

## Reading

No scored λ is quantum. The cavity-Λ curve bottoms near λ = 1.00 and is still 0.829 / 0.823 / 0.927 at n = 6 / 9 / 12. The product gap does clear 0.03 at λ = 1.50 and λ = 2.00 (about 0.12 and 0.087 per site), but those ground states stay above Λ = 0.87. λ = 0.50 is a classical bitstring (Z-peak 0.78–0.86, gap/n ≈ 0.001). The unscored scan on 8 instances, λ ∈ {0.25, 0.50, 1, 1.5, 2, 3, 4}, has the same shape: the lowest median Λ is 0.829 at n=6, λ=1, and λ ≥ 3 climbs back toward 1 as the state approaches |+⟩.

L=2 Walsh does not leave the Fourier mixer. Train loss moves by at most 0.001 after the warm start, and median-seed R_Fourier is 0.971–1.007 on the train sizes. That is above 0.98 at both n=6 and n=9 for λ = 0.50 and 1.00. At λ = 1.50 and 2.00 the n=6 L=2 ratios are 0.978 and 0.971, so the motif kill (R > 0.98 at n=6 and n=9 for every λ) does not fire. needs_edit does not apply either: the soft band is only for a λ that already clears Λ and the product gap.

L=4 is not the pass gate. Every L=4 Walsh polish hit maxiter=60 without an L-BFGS success flag, so those ratios are not certified minima (further polish can only lower the Walsh residual). Even the best of them, λ = 2.00 with n=12 R = 0.498, sits on a ground state with median Λ = 0.936.

## Decision rule

PASS if some scored λ has median cavity Λ ≤ 0.80 and product gap/n ≥ 0.03 at n=6, 9, and 12, and L=2 median-seed R_Fourier ≤ 0.85 at n12. needs_edit if quantumness holds but that R is soft. KILL if every scored λ fails quantumness, or if every λ has L=2 R > 0.98 at both n6 and n9.

**Verdict: KILL.**

## Scope

Wall time 120s. Train B=32, L-BFGS maxiter=60. Did not modify `Nov_3rd/settings.py`, did not write `qaoa_angles/`, and did not run or signal n21.

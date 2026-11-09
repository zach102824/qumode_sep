# Round-2 review (B1 IMAX, B2 BASIN, B3 N3X) and the next three entropy-only ideas (round 3)

**Scope.** Maximum-GS-entropy field \(B^*\) only. A = sites 0..⌊N/2⌋−1, matching the local stream. The b = 1/1.5 results are reference only. Nothing here has been launched; the parent coordinates the three existing workers.

**Context paper.** [arXiv:2204.02923](https://arxiv.org/abs/2204.02923). Its pair-phase construction is unrelated to the cavity-block M feature, and the paper findings remain separate from our hypotheses (see `IDEAS.md`).

**Rules kept.**
- SPSA only, for every candidate and for QAOA. Derivatives are diagnostics only.
- Shared \(U_{cZ}\); D in the middle of true Nov9th; original Nov3rd; QAOA p = L.
- Paired dev seeds; fully inclusive 128/1024 query ledgers.
- Approved D2–D5 simulation definitions; unconditioned F; no hardware claim.
- Validation and final seeds stay blind.

## 1. Round-2 outcomes (corrected numbers only; dev seeds 10000..10007)

| stream | question | outcome at B* |
|---|---|---|
| B1 IMAX (this PR, `results/IMAX/`) | Does the approved fixed imaginary-axis true displacement \(D(i\alpha)\) escape \(|+\rangle\), and does it beat QAOA? | **It escapes, but only shallowly.** It beats the real-axis control in every N=6/9 cell (8/8 wins, paired CI excluding 0). It never beats QAOA: 0/8, mean ΔF −0.20 to −0.50. It holds at N12 too: 720 runs, 0 failures, all certified at cutoff 32 vs 48 to 1.5e-13. Median η at L1, cap 1024 is +.133/+.062/+.046 for N6/9/12, so the escape shrinks with N. The strict-D8 diagnostic reaches .21/.15/.10, so leakage of about 1–3% costs something but is not the main limit. The **preregistered kill rule fires**: median η at N9, L1 is 0.062 < 0.25. Depth does not help (L2/L3 ≤ L1), and α is not saturated (max |α| ≤ .27 < .30). See `results/IMAX/INTERPRETATION_IMAX_M1.md`. |
| B2 BASIN (FT, PR #76) | Is there a reachable sub-product basin for **real** α? | Wide real-α/phase boxes and the frame start give no reliable escape: median ΔF vs QAOA −.30 to −.48, 0/8 everywhere. A minority of states do beat \(E_+\) by > 1e-3·N (the deepest: N6 L2 seed 10005, ΔE ≈ .14, F .341 vs \(F_+\) .264). So this is a finite-budget trap, **not** an expressivity theorem. |
| B3 N3X (GR, PR #77) | Does Nov3rd's L1 edge survive one matched, fully charged protocol? | **Yes at L1, cap 1024.** Median F for Nov3rd/QAOA: N6 .785/.640, N9 .710/.583, N12 .659/.521. Nov3rd wins 8/8 on E and F at every N; paired median ΔF is +.145/+.140/+.140. Fraction with F ≥ .7: 7/8, 5/8, 2/8. **The edge is weak at cap 128** (+.051/+.001/+.028) and **fades with depth**: at L3, cap 1024, the paired median ΔF is only +.069/+.050/+.027, the N12 marginal medians invert (Nov3rd .618 < QAOA .644), and at L3, cap 128, QAOA wins. |

**Consistent picture.** One mechanism explains all three outcomes; it is exact, and gap statements use the proper sector.

- **Real-α Nov9th.** With every SNAP phase at 0, all gates are real except the cost phase, so \(E(\gamma)=E(-\gamma)\) and the γ–α cross-curvature at \(|+\rangle\) vanishes. \(|+\rangle\) is a positive-curvature local minimum, and cold starts are captured there, though it is not a global ceiling (B2).
- **QAOA and Nov3rd.** Both leave \(|+\rangle\) invariant under their mixers (\(H_{\alpha\alpha}=0\)), so \(|+\rangle\) is a saddle for them. For Nov3rd, \(WSW=\exp[-i\theta(n^X_{\rm cav})]\) is a *per-cavity X-basis polynomial mixer*: linear X terms, with XX from \(s_2\) and XXX from \(s_3\). Checked to 6e-16. With \(s_2=s_3=0\) it is exactly QAOA's mixer \(e^{-i\beta\sum X}\) with \(\beta_l=-s_{l,1}/2\) (checked to 4e-16). Nov3rd is therefore "QAOA with one shared γ, K/M cost features, and intra-cavity XX/XXX mixer terms".
- **IMAX.** \(D(i\alpha)\) breaks the conjugation symmetry, giving a cost-coupled negative curvature, so it escapes. But a displacement does not leave \(|+\rangle\) invariant (\(H_{\alpha\alpha}\approx\) 18–36 at α-scale), which costs energy at every layer. That is why the basin is shallow and why extra layers do not help.
- **Gaps.** All 24 B* instances have E1 in the **odd** sector. The tiny global gaps (down to 2e-5 at N12 seed 10004, and .0013 at N6 seed 10002) are odd-sector. The relevant even-sector gaps are .029–1.52, and the smallest, .029, is at N12 seed 10001, which also has \(F_+\) = .055.

## 2. Next three ideas (round 3, B* only), ranked

### R3-1 N3-ABL: locate Nov3rd's B* edge by exact ablation (no decision needed)

- **Hypothesis.** Nov3rd beats QAOA at L1 because of its intra-cavity XX/XXX mixer terms and/or the K/M cost features, while its shared γ is what costs it at depth.
- **Arms.** All under GR's N3X_matched_v1 protocol, fully inclusive, paired seeds; no new physics, all are restrictions of the original Nov3rd:
  - (a) full Nov3rd;
  - (b) \(s_3=0\);
  - (c) \(s_2=s_3=0\): exactly QAOA with shared γ plus K/M;
  - (d) (c) with \(\theta_K=\theta_M=0\): QAOA with one shared γ;
  - QAOA p = L.
- Run N6/9/12, L1–3, caps 128/1024.
- **Read rules.**
  - (a) − (c) > 0.05 paired at L1: the multi-X mixer terms carry the edge.
  - (c) − (d) > 0.05: the K/M features do.
  - (d) < QAOA by more than 0.05 at L3: shared γ explains the depth fade.
- **Why it is worth it.** It is the only stream with a real same-L advantage, and the ablation converts it into a mechanism. It costs about as much as B3.

### R3-2 GROW: layerwise-continuation SPSA for depth (no decision needed)

- **Hypothesis.** The depth fade (Nov3rd L3 ≈ QAOA, IMAX L2/L3 ≤ L1) comes from SPSA conditioning in more dimensions within 1024 queries, not from the ansatz.
- **Protocol.**
  - Within the same inclusive 1024 queries: spend 25% at L = 1, then append layer 2 initialized near identity (α = 0 or s = 0) and spend 25%, then layer 3 and spend 50%.
  - The shared \(U_{cZ}\) parameters carry over.
  - Every query is charged. Selection is the lowest counted candidate.
  - The identical schedule is applied to QAOA (new β = 0, γ = 0), Nov3rd and IMAX.
- **Read rules.**
  - Nov3rd L3 GROW beats Nov3rd L3 cold by more than 0.05 paired, and beats QAOA L3 GROW: depth is recoverable.
  - Otherwise, the L1 edge is the ceiling of the shared-γ design.
- **Fairness.** QAOA gets the same continuation benefit, so any remaining advantage is not a schedule artifact.

### R3-3 IM-WIDE: close or open the Nov9th line at B* (no new decision; uses the approved D(iα))

- **Hypothesis.** IMAX's cold optimum is the shallow basin next to \(|+\rangle\). A deeper basin may sit farther out (larger γ, |s| ~ π), since the negative curvature exists. B2 found real-α states beating \(E_+\) only in a minority.
- **Design.** B2's wide-start successive-halving audit applied to IM and IM_B: 64 starts, with 256 queries per start, then 1024, then 4096 for the top 8. Energy only, no GS oracle, α within 0.30. N6/9, L1–2. It is a diagnostic; its numbers are never reported as method performance.
- **Read rules.**
  - No IM point reaches η ≥ 0.25 at N9 L1 even with the diagnostic budget: **close the Nov9th line at B*** and give its worker slot to R3-1 or R3-2.
  - A deep basin exists: freeze an instance-independent start recipe from it (dev only, with configurations charged). Rerun the standard 1024 protocol as a labeled initialization variant.

## 3. What not to do next

- Do not spend more on real-α Nov9th cold starts: three streams and every variant are null.
- Do not compare marginal medians when the claim is a paired one; report both (B3's N12 L3 case).
- No N15 optimization until R3-1 or R3-2 shows an edge that survives at N12.

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
| B1 IMAX (this PR, `results/IMAX/`) | Does the approved fixed imaginary-axis true displacement \(D(i\alpha)\) escape \(|+\rangle\), and does it beat QAOA? | **It escapes, but only shallowly.** It beats the real-axis control in every N=6/9 cell (8/8 wins, paired CI excluding 0). It never beats QAOA: 0/8, mean ΔF −0.20 to −0.50. It holds at N12 too: 720 runs, 0 failures, all certified at cutoff 32 vs 48 to 1.5e-13. Median η at L1, cap 1024 is +.133/+.062/+.046 for N6/9/12, so the escape shrinks with N. The strict-D8 diagnostic reaches .21/.15/.10, so leakage of about 1–3% costs something but is not the main limit. The **preregistered kill rule fires**: median η at N9, L1 is 0.062 < 0.25. α is not saturated (max |α| ≤ .27 < .30). The depth fade is an **optimization failure**, not expressivity. L1 embeds exactly in L3 (γ/3, layers 1–2 identity; error ≤ 8e-15), yet the independently optimized L3 energy is worse than the embeddable L1 in 5/8, 7/8 and 4/8 seeds at N6/9/12. For IM_B it is worse in 5/8, 8/8 and 7/8. QAOA L3 always beats its L1 (0/8 worse). See `results/IMAX/depth_nesting_audit.json` and `INTERPRETATION_IMAX_M1.md`. |
| Local R3 (Nov3rd vs QAOA, local continuation recipe; **do not merge with GR medians**) | Does the Nov3rd edge hold at larger N, deeper circuits and new seeds? | Old seeds 10000–10003, cumulative 3072, median F Nov3rd/QAOA: **L1** N9 .697/.572, N12 .621/.461, N15 .681/.558. **L3** N9 .860/.812, N12 .775/.770, N15 .729/.749. A higher median F did not mean a consistent energy advantage. New dev seeds 10004–10007 at N15 under the frozen two_long recipe (actual cost 5120 per cell): **L1 .629/.507, Nov3rd wins 4/4 on E and F**, but L3 .666/.692 with only 2/4 ≥ .7. So the old-seed L3 claim does **not** generalize, while the L1 edge does. Depth-nesting: Nov3rd L1 embeds exactly in L3 (5.1e-15), and the embedded L1 beats the optimized L3 (−14.145 vs −14.043). That is an optimization failure, the same pattern as IMAX. Local is testing charged L1→L3 transfer for Nov3rd next. |
| B2 BASIN (FT, PR #76) | Is there a reachable sub-product basin for **real** α? | Wide real-α/phase boxes and the frame start give no reliable escape: median ΔF vs QAOA −.30 to −.48, 0/8 everywhere. A minority of states do beat \(E_+\) by > 1e-3·N (the deepest: N6 L2 seed 10005, ΔE ≈ .14, F .341 vs \(F_+\) .264). So this is a finite-budget trap, **not** an expressivity theorem. |
| B3 N3X (GR, PR #77) | Does Nov3rd's L1 edge survive one matched, fully charged protocol? | **Yes at L1, cap 1024.** Median F for Nov3rd/QAOA: N6 .785/.640, N9 .710/.583, N12 .659/.521. Nov3rd wins 8/8 on E and F at every N; paired median ΔF is +.145/+.140/+.140. Fraction with F ≥ .7: 7/8, 5/8, 2/8. **The edge is weak at cap 128** (+.051/+.001/+.028) and **fades with depth**: at L3, cap 1024, the paired median ΔF is only +.069/+.050/+.027, the N12 marginal medians invert (Nov3rd .618 < QAOA .644), and at L3, cap 128, QAOA wins. |

**Consistent picture.** One mechanism explains all three outcomes; it is exact, and gap statements use the proper sector.

- **Real-α Nov9th.** With every SNAP phase at 0, all gates are real except the cost phase, so \(E(\gamma)=E(-\gamma)\) and the γ–α cross-curvature at \(|+\rangle\) vanishes. \(|+\rangle\) is a positive-curvature local minimum, and cold starts are captured there, though it is not a global ceiling (B2).
- **QAOA and Nov3rd.** Both leave \(|+\rangle\) invariant under their mixers (\(H_{\alpha\alpha}=0\)), so \(|+\rangle\) is a saddle for them. For Nov3rd, \(WSW=\exp[-i\theta(n^X_{\rm cav})]\) is a *per-cavity X-basis polynomial mixer*: linear X terms, with XX from \(s_2\) and XXX from \(s_3\). Checked to 6e-16. With \(s_2=s_3=0\) it is exactly QAOA's mixer \(e^{-i\beta\sum X}\) with \(\beta_l=-s_{l,1}/2\) (checked to 4e-16). Nov3rd is therefore "QAOA with one shared γ, K/M cost features, and intra-cavity XX/XXX mixer terms".
- **IMAX.** \(D(i\alpha)\) breaks the conjugation symmetry, giving a cost-coupled negative curvature, so it escapes. But a displacement does not leave \(|+\rangle\) invariant (\(H_{\alpha\alpha}\approx\) 18–36 at α-scale), which costs energy at every layer. That is why the L1 basin is shallow. The failure of extra layers to help is separate: it is SPSA failing to reach the embedded L1 point at L3, as it does for Nov3rd.
- **The common finding across streams:** the depth fade for both Nov3rd and IMAX is an optimization failure with an exact certificate. The only robust same-L advantage over QAOA is Nov3rd at L1, which holds 8/8 at N6/9/12 (GR) and 4/4 on new seeds at N15 (local).
- **Gaps.** All 24 B* instances have E1 in the **odd** sector. The tiny global gaps (down to 2e-5 at N12 seed 10004, and .0013 at N6 seed 10002) are odd-sector. The relevant even-sector gaps are .029–1.52, and the smallest, .029, is at N12 seed 10001, which also has \(F_+\) = .055.

## 2. Next three ideas (round 3, B* only), with worker assignment

These apply to all three:
- B* only, using the frozen interior B* instances and A = sites 0..⌊N/2⌋−1.
- Approved D2–D5 definitions; imaginary-axis \(D(i\alpha_l)\) is authorized with one real amplitude per layer on a fixed axis and no axis parameter.
- SPSA only, with a fully inclusive ledger in which every energy call is charged, including any transfer source. QAOA p = L.
- Paired dev seeds 10000–10007 (N15: 10004–10007 have already been used locally as dev). Unconditioned F.
- Report paired ΔF and ΔE alongside the marginal medians.
- No more real-α cold starts.

### R3-A (worker GR): N3-ABL, exact ablation of Nov3rd's L1 edge

- **Why.** Nov3rd at L1 is the only robust same-L advantage over QAOA: 8/8 at N6/9/12 in GR's matched run, and 4/4 on new N15 seeds locally. Turning it into a mechanism is the measurable next step.
- **Arms** (GR's N3X_matched_v1, caps 128/1024; every arm is an exact restriction of the original Nov3rd):
  - (a) full;
  - (b) \(s_3=0\);
  - (c) \(s_2=s_3=0\): exactly QAOA with one shared γ plus K/M, since \(\beta_l=-s_{l,1}/2\) to 4e-16;
  - (d) (c) with \(\theta_K=\theta_M=0\): QAOA with a shared γ;
  - QAOA p = L.
- **Scope.** L1 at N6/9/12/15, plus L3 at N9/12 as context only.
- **Read rules.**
  - (a) − (c) ≥ 0.05 paired at L1: the intra-cavity XX/XXX mixer terms carry the edge.
  - (c) − (d) ≥ 0.05: the K/M features carry it.
  - (d) ≈ QAOA L1: shared γ costs nothing at L1.
  - If (b) is no worse than (a) and better at cap 128, adopt (b) as a cheaper recipe; it has fewer parameters.

### R3-B (worker LE, this branch): IM-DEPTH, charged L1→L_target depth transfer for IMAX, with QAOA on the same schedule

- **Why.** The IMAX depth fade is a certified optimization failure: L3 ends above its exactly embeddable L1 in 50–88% of seeds for true D, and 63–100% for strict D8. Local is doing the Nov3rd transfer, so this covers the Nov9th side without duplication.
- **Protocol.**
  - Within 1024 inclusive queries: 2 starts × 256 at L = 1.
  - The best L1 point is embedded exactly in L_target (γ/3 at L3, or γ/2 at L2; earlier layers set to α = s = 0, the last layer carrying the L1 mixer). The embedded point is charged as a candidate.
  - Then 2 SPSA starts × 256 at L_target: from the embedded point, and from the embedded point plus a small perturbation.
  - QAOA uses the identical schedule (its L1 embeds trivially with the extra layers' angles set to 0). It is compared against the cold 1024 runs already in `results/IMAX`.
  - Arms: IM, IM_B (diagnostic), QAOA. N6/9/12, L2/L3, cap 1024, plus a cap-128 version that scales each share proportionally.
- **Read rules.**
  - By construction the selected E is ≤ the embedded L1 E. Success means IM L3 transfer beats IM L1 cold by ≥ 0.05 paired median F at N9.
  - The IM − QAOA paired ΔF under identical schedules must be reported. With IMAX's L1 η of only .06, a same-L QAOA win is not expected. The point is whether depth becomes usable.
  - If the transfer gives < 0.02 gain over L1, close the IMAX depth question.

### R3-C (worker FT): IM-WIDE, a wide-start basin audit for \(D(i\alpha)\) that closes or opens the Nov9th line at B*

- **Why.** IMAX's cold optimum is shallow (η .13/.06/.05). FT already has the B2 successive-halving tooling and NF32/48 certification, so this is cheap for FT.
- **Design.**
  - B2's audit with the axis switched to the approved imaginary one: IM and IM_B, 64 starts with s ~ U(−π, π)^{3L}, γ ~ U(−1, 1), θ ~ U(−1, 1), α within 0.30.
  - Successive halving: 256 queries for every start, then 1024, then 4096 for the top 8.
  - Energy only, no GS oracle. N6/9, L1–2. It is a diagnostic; its numbers are never reported as method performance.
- **Read rules.**
  - No IM point reaches η ≥ 0.25 at N9 L1: **close the Nov9th line at B***, and FT moves to supporting R3-A at N15 or R3-B at N12.
  - A deep basin exists: freeze an instance-independent start recipe from it (dev only, configurations charged), then rerun the standard 1024 protocol as a labeled initialization variant.

### Priority if only two workers are available
R3-A first, then R3-B. R3-C is the cheapest and should still run, because it decides whether the Nov9th line continues at all.

## 3. What not to do next

- Do not run more real-α Nov9th cold starts: every stream and variant is null at both b = 1/1.5 and B*.
- Do not compare marginal medians when the claim is paired; report both (B3's N12 L3 case).
- Do not merge GR's matched medians with the local continuation-recipe medians.
- Charge every depth-transfer source fully, and do not replace the reported cold L3 curves.
- No N15 optimization from the clouds unless R3-A needs N15 at L1 (seeds 10004–10007 are already dev-used locally).

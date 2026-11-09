# Nov9_SK_SPSA round 2: three next ideas, B* (maximum-GS-entropy field) ONLY

**Scope (user, Nov 9 19:05 Shanghai).** Every new optimization experiment uses only the maximum-GS-entanglement-entropy field \(B^*\). The b = 1/1.5 results are a finished reference and no further b = 1/1.5 runs are scheduled. Nothing here has been launched; the parent coordinates the three existing workers.

## Constraints common to all three ideas

- **Instance definition.**
  - SK J as before, from `default_rng(seed)` with variance 1/N.
  - **One fixed balanced spin bipartition** for the whole study. Use the local stream's choice and record its site list in the protocol. My diagnostics assumed \(A=\{0..\lfloor N/2\rfloor-1\}\) (LSB = site 0) and reproduced the local smallest gap (.0013, N6 seed 10002). Confirm that the lists agree.
  - \(B^*\) is the per-instance refined argmax of \(S_A(b)\) for the exact GS. Use a coarse grid, then bounded refinement. The peak must be interior: a peak on a grid edge invalidates the instance. Assume no criticality.
  - Checks per instance:
    - GS even under global spin flip, and unique. For b > 0, H is stoquastic and irreducible, so the GS is unique and positive.
    - Report \(E_1-E_0\) and the even-sector gap \(\Delta_{\rm even}\). When E1 is odd, a parity-preserving circuit does not feel the small gap.
    - Report \(F_+\) and \(S_A\).
  - \(B^*\) is computed from the exact GS. It defines the instance, and the circuit sees it only through the field term of H. It is never a circuit feature.
- **Methods.**
  - SPSA only: no BFGS, no natural gradient. Derivatives are allowed as diagnostics only.
  - Shared \(U_{cZ}\); D in the middle of true Nov9th.
  - Original Nov3rd.
  - QAOA p = L with 2p independent angles.
- **Budget.** 1024 inclusive queries with 4 starts, plus the cap-128 curve. Every energy call that influences the output is charged, including the initial and diagnostic calls behind GR's 1028.
- **Seeds, definitions, reporting.**
  - Paired dev seeds. Validation and final stay blind until the recipe is frozen.
  - Approved D2–D5 simulation definitions; NF ≥ 32 or a converged ladder (FT's NF16 contraction lesson).
  - Unconditioned F; no hardware claim.
  - Every table reports F, \(F_+\), η, paired ΔF and ΔE against QAOA, \(p_{\rm leak}\), the gaps and \(S_A\).

## Evidence carried in (corrected numbers only)

- **LE, b = 1/1.5 reference.**
  - Real-α Nov9th stays at \(F_+\) in every arm: Uz and U0, \(s_3\) on and off, alternating, same-sign and frame starts, strict D8. This holds at N6/9/12 with 0/8 wins.
  - Leakage, cutoff and extension are ruled out as causes.
  - See `results/LE/INTERPRETATION_LE_M1.md`.
- **FT (NF32 replay; not the abort-biased metrics).** The plus-stall is not rescued, and fine-tune transfer is worse than cold. Cold-start slopes are several units, but the downhill direction returns to the product state.
- **GR.** Encoding and assignment changes move F by about 0.001. The GR and local Nov3rd numbers come from different protocols; do not merge them.
- **Local, \(B^*\)** (4 matched dev J per N; Nov3rd, QAOA, Nov9th):
  - Nov9th F ≈ .27–.31, at the product level.
  - Nov3rd L1: .74/.67/.59/.66 at N6/9/12/15.
  - QAOA L1: .64/.57/.46/.56; it generally overtakes Nov3rd by L3.
  - Gaps .0013–.12.
- **Mechanism (exact; holds at \(B^*\); `bstar_curvature.json`).**
  - With SNAP phases s = 0, all real-α Nov9th gates are real except the cost phase. So \(E(\gamma)=E(-\gamma)\) for any α (verified to 0.0), and the γ–α cross-curvature at \(|+\rangle\) vanishes.
  - QAOA's mixer and Nov3rd's \(WSW\) leave \(|+\rangle\) invariant (\(H_{\alpha\alpha}=0\)), so their \(|+\rangle\) is a saddle. A displacement moves \(|+\rangle\) (\(H_{\alpha\alpha}>0\)), so for real α, \(|+\rangle\) is a positive-curvature local minimum.
  - Full-Hessian minimum eigenvalue at \(|+\rangle\) for \(B^*\), N6/N9, 4 seeds each:

    | circuit | L = 1 | L = 2 |
    |---|---|---|
    | real α | 0 | 0 |
    | imaginary-axis \(D(i\alpha)\) | −2.0 to −7.9 (γ component 0.42–0.67) | −4.0 to −12.0 |
    | QAOA | −7.2 to −28 | −11.6 to −45 |

  - This explains the nulls from cold starts. It does **not** prove an expressivity ceiling for real α.

## Idea B1 (needs a user decision): imaginary-axis true displacement at \(B^*\), IMAX

- **Change.** \(D(i\alpha)=e^{i\alpha(a^\dagger+a)}\): one real control per layer, a true displacement driven at quadrature phase. D stays in the middle and \(U_{cZ}\) stays shared. It is a definition change from "real α" (a restricted D7), so it needs explicit approval and must be labeled IMAX, never Nov9th-real.
- **Why at \(B^*\).** It is the only Nov9th-type circuit with a cost-coupled escape from \(|+\rangle\), and the escape is stronger at \(B^*\) than at b = 1. \(F_+\) is low there (.16–.40), so escaping matters most. No SPSA run has been done yet.
- **Arms.** IMAX under Uz; IMAX under U0 (sensitivity); IMAX strict D8 (diagnostic); a fixed displacement phase of π/4 (labeled sensitivity); paired QAOA and Nov3rd. N6/9 at L1–3, then N12/15 only where convergence is measured.
- **Kill and read rules.**
  - At N9, L1, median η < 0.25: stop.
  - Strict D8 good but true D failing: leakage is back, so rerun the leakage controls.
  - The target is same-L F above QAOA. QAOA is weak at \(B^*\) L1 (.46–.64), so this is a real test rather than a trivial threshold.

## Idea B2 (no decision needed): reachable-basin and expressivity audit of approved real-α Nov9th at \(B^*\), BASIN

- **Purpose.** Separate cold-start capture from expressivity: FT asked not to infer a ceiling from failure, and recommended looking for a reachable low-energy basin with a real-α-plus-phase initialization. This is a diagnostic audit; its numbers are never reported as Nov9th performance.
- **Design.**
  - Energy objective only, with no GS oracle.
  - Wide symmetry-broken starts: \(s\sim U(-\pi,\pi)^{3L}\), γ ~ U(−1, 1), θ ~ U(−1, 1); α within D9 (0.30), plus labeled α_max = 0.6/1.0 sensitivity arms.
  - The analytic π/2 frame start is included as one start family. It is a known null at b = 1, where its curvature is cavity-local with a γ component of about 0.003.
  - 64 starts with successive halving (256 queries for every start, then 1024, then 4096 for the top 8), all SPSA. N6/9, L1–3.
  - Record the best \(E-E_+\), F at the selected point, leakage, and the basin parameters.
- **Read rule.**
  - No point beats \(E_+\) by more than \(10^{-3}N\): report "no basin under budget X", which is evidence but not proof.
  - A basin exists: freeze an instance-independent start recipe from it (dev only, configurations charged). Rerun the standard 4-start 1024 protocol from that recipe as a labeled initialization variant; the ansatz is unchanged.

## Idea B3 (no decision needed): Nov3rd's \(B^*\) L1 edge under a single matched protocol, N3X

- **Why.** Locally, Nov3rd at \(B^*\) is the only non-QAOA circuit with a same-L advantage (L1: .59–.74 vs QAOA .46–.64), but QAOA overtakes by L3. Nov3rd's \(WSW\) mixer leaves \(|+\rangle\) invariant, the same property real-α Nov9th lacks, so it is the positive control for the mechanism. GR showed the edge is quality, not speed (at b = 1: .973 vs .933 at cap 1024, but .863 vs .928 at cap 128), and the GR and local protocols differ (monitoring 0 vs 340 of 1024).
- **Design.**
  - One protocol for every arm: 4 starts, 1024 inclusive queries, all calls charged, no monitoring split, plus the cap-128 curve.
  - Nov3rd and QAOA paired at L1–3, N6/9/12/15, on 8 dev seeds.
  - Add the \(|+\rangle\) 2×2 curvature diagnostic for both.
  - Report the crossover depth, the budget curve, and η against the gap and \(S_A\).
- **Read rule.**
  - L1 advantage ≥ 0.05 median F on ≥ 6/8 seeds at N12/15: the result is robust, and Nov3rd at L1 becomes the reference that any Nov9th variant (B1) must match.
  - The advantage disappears under the matched protocol: it was a protocol artifact; report that.

## Ranking

| rank | idea | needs a user decision | expected gain | main risk |
|---|---|---|---|---|
| 1 | B1 IMAX | yes (displacement phase convention) | highest for a Nov9th-type circuit | it may be QAOA-like without beating QAOA |
| 2 | B3 N3X | no | firm reference, and it tests the mechanism | the edge may vanish under the matched protocol |
| 3 | B2 BASIN | no | decides the real-α question | diagnostic only, with no method gain |

B2 and B3 can start immediately; B1 waits for the decision. If B1 is refused, its worker joins B2's wide-start search.

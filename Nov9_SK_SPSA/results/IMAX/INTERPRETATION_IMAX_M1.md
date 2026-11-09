# R2-B1 IMAX milestone 1: interpretation (B* only, dev seeds 10000..10007)

**Status.**
- **Approvals.** The fixed imaginary-axis true displacement \(D(i\alpha)=e^{i\alpha(a^\dagger+a)}\) was approved for testing by the user (Nov 9 19:10 Shanghai). The run uses the approved D2–D5 simulation definitions. It is **not** a hardware result.
- **Prereg.** `experiments/IMAX/PREREG_IMAX_M1.json` (sha256 aad8273c…), frozen before any fit. Conformance tests T17–T26 all pass (`TESTS.json`).
- **Runs.** 720 runs (N6/9/12 × L1–3 × 5 arms × caps 128/1024 × 8 seeds), 0 failures. Every run used exactly 128 or 1024 counted queries: calibration, perturbation and candidate calls are all charged, with no monitoring and no uncounted calls.
- **Cutoff.** Every leaky selected point is certified at cutoff 32 against 48: max \(|\Delta E|\) 1.5e-13, \(|\Delta F|\) 6e-15, \(|\Delta p_{\rm leak}|\) 2e-14, \(|\lVert\psi\rVert-1|\) ≤ 2e-14. Cutoffs 16 and 24 also agree to 2e-12.
- **Fidelity.** F is unconditioned, against the exact GS, computed post hoc only. With 8 samples there is no target claim.

Tables: `RESULTS_M1.md`, `summary_M1.json`. Plots: `medF_vs_L.png`, `paired_F_scatter.png`, `leak_vs_NF.png`.

## Instances

There are 24 B* instances, all valid. The checks are: interior refined peak; unique, Perron-positive, flip-even GS; residual < 1e-9.

- Median \(F_+\) is .309/.291/.206 at N6/9/12.
- **E1 is odd in every instance.** The small global gaps (2e-5 at N12 seed 10004, .0013 at N6 seed 10002) are odd-sector.
- The even-sector gaps range from .029 (N12 seed 10001) to 1.52.

## Results at cap 1024 (median F; paired mean ΔF with 95% bootstrap CI)

| N | L | QAOA | IM \(D(i\alpha)\) | RE \(D(\alpha)\) | IM_B (strict, not true D) | IM − QAOA | IM − RE | IM median η |
|---|---|---|---|---|---|---|---|---|
| 6 | 1 | .639 | .401 | .309 | .450 | −.202 [−.235, −.169], 0/8 | +.101 [+.083, +.120], 8/8 | +.133 |
| 6 | 2 | .824 | .416 | .310 | .461 | −.383, 0/8 | +.085, 8/8 | +.122 |
| 6 | 3 | .849 | .402 | .313 | .422 | −.438, 0/8 | +.081, 8/8 | +.114 |
| 9 | 1 | .581 | .329 | .291 | .381 | −.242 [−.268, −.217], 0/8 | +.048 [+.032, +.065], 8/8 | +.062 |
| 9 | 2 | .726 | .325 | .291 | .348 | −.412, 0/8 | +.043, 8/8 | +.063 |
| 9 | 3 | .801 | .317 | .291 | .356 | −.503, 0/8 | +.021, 8/8 | +.024 |
| 12 | 1 | .519 | .230 | .206 | .267 | −.234 [−.273, −.186], 0/8 | +.034 [+.025, +.041], 8/8 | +.046 |
| 12 | 2 | .671 | .224 | .206 | .241 | −.410, 0/8 | +.025, 8/8 | +.028 |
| 12 | 3 | .721 | .217 | .205 | .253 | −.496, 0/8 | +.021, 8/8 | +.023 |

E is paired the same way: IM loses on E 0/8 in every cell, and RE sits at \(E_+\) to within ±0.01 of the gap.

**Cost curves.** Cap 128 against cap 1024:
- QAOA is mostly converged by 128 queries (for example N9 L1 .583 → .581, N12 L3 .687 → .721).
- IM gains a little (N6 L1 .374 → .401; N12 L1 .217 → .230).
- RE is flat at \(F_+\).

**Tails.** IM never reaches F ≥ .5 at N ≥ 9; at N6 it reaches .5 for at most 2/8 seeds (L1). QAOA reaches F ≥ .5 for 7/8 seeds at N9 L1 and 7/8 at N12 L3.

## Verdicts (preregistered reading rules)

1. **The kill rule fires.** At N9, L1, cap 1024, IM's median η is 0.062 < 0.25, so IMAX fails its first gate.
2. **The prediction is confirmed qualitatively but not quantitatively.** The imaginary axis does escape \(|+\rangle\): IM beats RE 8/8 in every N/L cell, with CIs excluding 0, while RE stays at \(F_+\) with γ → 0.001 (the predicted trap). The predicted negative curvature at \(|+\rangle\) gives only a shallow basin. The escape shrinks with N: median η is .133/.062/.046 at L1 for N6/9/12.
3. **Leakage is secondary.** \(p_{\rm leak}\) is about 1–3% for IM. IM_B, with no leakage, does better (η .21/.15/.10 at L1), but it is also < 0.25 and far below QAOA. Per the rule ("leakage-limited only if IM_B η ≥ 0.5"), IMAX is not leakage-limited.
4. **Not extension-dependent.** The median \(|F_{\rm Uz}-F_{\rm U0}|\) at identical parameters is ≤ 3.6e-4.
5. **No same-L advantage.** Every IM − QAOA CI is entirely negative.
6. **Depth does not help.** IM at L2/L3 is ≤ L1. α is not saturated (max |α| ≤ .27 < .30, so D9 is not binding), and γ shrinks as L grows.

**Mechanism.** \(D(i\alpha)\) breaks the real-gate conjugation symmetry, which supplies the cost-coupled negative curvature. But it does not leave \(|+\rangle\) invariant: \(H_{\alpha\alpha}\) ≈ 18–36 per α², whereas QAOA's mixer and Nov3rd's \(WSW\) have 0. Every layer of displacement therefore pays an energy cost on the reference state, which caps the basin depth and is consistent with no gain from depth. This is a diagnosis of the cold-start optimum under 1024 queries, not an expressivity theorem.

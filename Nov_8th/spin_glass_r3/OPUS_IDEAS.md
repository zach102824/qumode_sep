# Round 3 ideas from the second (and last) Opus 5.5 medium consultation

One consultation (`claude-opus-5-5-medium`, read-only, agent id `bc-abec95c3-73f0-5016-8429-4c0636cefb4a`).
Condensed faithfully below. Items marked "checked" were verified before the protocol was frozen.

## 1. Critique of round 2

No correctness bug was found in the R2 simulator or selection logic. The R2 "11/12 wins" is weaker evidence than it looks.

Checked clean:
- The box does not bind. No selected R2 cell has a coordinate on a bound; the largest diagonal coefficient uses
  about 9-10% of the box for shared families and 17% for QAOA.
- Gradients agree with finite differences (<= 4e-8). No start reached the cap. Tight L6 polish moved energy <= 1e-10,
  although selected KM states had projected gradients near 1e-4 before polish.
- QAOA's mixer is plain sum X (`w1 == wX`).
- No preserved round-1 state was selected; round-1 seeds were given to all four families.

Confounds that could inflate the KM advantage:
1. Effort asymmetry. KM inherits K's search through nesting, and K inherits nodata's. KM used 3 to 3.5 times as many
   evaluations as QAOA. QAOA's L6 F_sub equals round 1 in all 12 instances, which suggests saturation, but extra QAOA
   effort was never tested. KM's landscape is search-sensitive (N9 k0 0.024 -> 0.908; N12 k3 0.001 -> 0.596).
2. Mixer expressivity. Checked: `shell_weights` makes w2 the sum of X_a X_b over pairs inside each three-site cavity
   and w3 the product X_a X_b X_c, so KM carries two- and three-body mixers while QAOA carries one-body sum X. The
   comparison is "KM package vs standard QAOA", not a test of the K and M features alone. Per-layer gamma with the same
   shells was never run.
3. Equal depth is not equal parameters (21 vs 12 at L6). At equal parameter count (KM L3 vs QAOA L6) QAOA has higher
   mean F_sub at every N. Report as a secondary view only.
4. N12 k1 counted as a win (0.018 vs 0.001) although both families failed. A floor/tie rule is needed.
5. The convergence flag is nearly vacuous: children of the same parent (interp, time-rescaled, sigma 0.1) fall back
   into the same basin, so "two starts agree" is not evidence of a global optimum.
6. k=0..3 were R1 discovery instances; round-1 seeds came from family-specific searches; the L1 grid is deterministic,
   so a repeat of the procedure is not independent.
7. Replay reused the same `hc`, `apply_X_sum` and dense eigh; it was not independent.
8. Parity: the circuit is confined to the even sector. If a new odd-N instance has a doublet splitting above 1e-6,
   F_sub is structurally zero for every family. Such instances must be predeclared ties, not dropped.
9. An absolute +0.05 margin is impossible near F = 1; relative infidelity is the right scale.

## 2. Ranked ideas for the robustness round

1. Fresh iid instances as the primary set (k = 4..27, 24 per N); report k = 0..3 separately.
2. Three independent repeats per instance with different random, grid-offset, perturbation and ramp-jitter seeds;
   common random numbers across families within a repeat.
3. Drop round-1 and round-2 seeds so every family has identical provenance.
4. Budget control "QAOA-heavy": 3x the L1 random starts and 2x perturbed continuation draws per parent.
5. Paired statistics over instances on log infidelity ratio, with predeclared ties and floors.
6. Independent replay: Pauli-string `scipy.sparse.kron` construction, `expm_multiply`, `eigsh`.
7. Per-start lineage and projected-gradient logging; count KM selections with a nested-seed ancestor.

## 3. Recommended protocol (adopted with the details frozen in PREREG.json)

Instances from `SeedSequence([20261115, N, k])`; k = 0..3 must reproduce the R1 manifest hashes. Repeats r = 0, 1, 2
with grid offset u = pi (j + delta_r)/16, c0 grid jitter +-0.05, and new seed roots for random, perturbation and ramp
draws. R2 box, beam 3, nested seeds and continuation menu unchanged; cap 2000. Selection is energy only: per repeat,
and best over repeats (primary). Win if rho = log[(1 - F_Q)/(1 - F_K)] > log 1.1 and dF > 0.005; mirror for loss;
otherwise tie; tie "both failed" if max F < 0.02; tie "unreachable" if the 1e-6 window has no even-sector state.
One-sided exact sign test (ties excluded), Wilcoxon, bootstrap CIs over instances, Holm over the three N at L = 6.
Claim at N requires (a) Holm p < 0.05 at L = 6, (b) bootstrap lower bound of median rho > 0 at L = 6, (c) L = 5 sign
p < 0.05 with rho CI lower bound > 0, (d) (a) and (b) survive in at least 2 of 3 per-repeat selections and with
QAOA-heavy replacing QAOA. Joint wins (fidelity win and E_KM <= E_Q + 1e-6) reported separately from fidelity-only
wins. Reliability reported separately from physics. Bound coverage reported.

## Untested future ideas (documented, not run)

- KM with per-layer gamma_l (shared theta shape).
- QAOA plus the w2/w3 cavity shells with per-layer gamma: the clean ablation of confound 2.
- KM without the w2/w3 shells.
- Equal compiled two-qubit-gate comparison.
- Packing-label permutation test.
- N = 15 with sparse simulation; other field strengths and signs; +-J disorder.
- Warm-start or adaptive-bias QAOA from He, Kemper and Freericks.

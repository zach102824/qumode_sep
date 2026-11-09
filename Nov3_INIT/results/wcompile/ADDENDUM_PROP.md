# Addendum: PR109 propagator was repeated truncation, not oscillator evolution

Historical **WCOMP_FAIL_BUDGET** numbers are preserved. They are not evidence that native SNAP+D compilation is impossible. No refit. SK ledger 2818048. Historical compiler evals 2048 remain recorded.

## What PR109 did

Each displacement was `P_nf D_{nf+32}(α) P_nf` with nf=16 (replay 32). That block is non-unitary. Applying it after every D **throws away pad amplitudes before the next gate**. That is implicit repeated projection, not a sequence on one oscillator. SNAP pad phase 0 was already unitary.

Audit check: `workspace_map` at nf=16 matches PR109 `compiled_map` (Frobenius 0.00e+00).

## 0.402 vs 0.672

PR109 `col_norm` is **mean ‖ψ‖²** (retained probability), not Euclidean ‖ψ‖. Mean Euclidean norm was 0.6265.

- mean p_total = 0.4020 (the 0.402 figure)
- mean p_code = 0.3277
- mean p_pad in 8..15 = 0.0743
- mean p_lost = 0.5980 (discarded by the three projections)
- old p_leak = 1-p_code = 0.6723 = pad + lost

F_avg / F_worst / F_HS used unnormalized overlaps against the padded isometry T. They are not postselected state fidelities. When ‖ψ‖²≈0.40 they cannot be read as physical gate error.

## Identity vs selected on the actual objective

Start 0 was the identity compile and **was evaluated** (not excluded). NF=16 broken propagator: identity L = 2.000, selected polyak64 L = 0.822. Selected **beats** identity on L because shrinking ‖V‖ lowers ‖V−T‖_F even though F_avg 0.117 **loses** to identity 0.125.

Common workspace NF=96 (no refit, unit norm): identity L = 2.0000, selected L = 1.4342. Selected still beats identity on L (True) because tr(W)=0 so identity has HS overlap 0, while the pulse has a globally alignable positive overlap. F_avg 0.112 still **loses** to identity 0.125. The extra drop from L=1.434 (unitary workspace) to L=0.822 (PR109) is the nonunitary projection shrinking ‖V‖.

## Frozen-pulse replay on a common workspace (32, 64, 96)

Same `pulse.json`. D defined as restrict(exp_{W+32}) onto W, then the whole (D,SNAP)^3 stays in C^W. No renormalization.

| W | mean p_total | mean p_code | mean p_pad | mean p_lost | mean p_boundary | F_avg | F_HS | L |
|---|---|---|---|---|---|---|---|---|
| 32 | 0.583765 | 0.3262 | 0.2576 | 4.162e-01 | 4.600e-02 | 0.1120 | 0.0800 | 1.0179 |
| 64 | 0.989931 | 0.3262 | 0.6637 | 1.007e-02 | 2.621e-02 | 0.1120 | 0.0800 | 1.4241 |
| 96 | 1.000000 | 0.3262 | 0.6738 | 7.009e-13 | 3.393e-11 | 0.1120 | 0.0800 | 1.4342 |

Full-space column overlap (pad-embed, one global phase): 32 vs 64 mean |⟨col⟩|² 0.3658; 64 vs 96 0.980317; 32 vs 96 0.3658.
Code-block Frobenius 32 vs 96: 3.547e-05; 64 vs 96: 5.753e-14.
F_avg is identical on the ladder (0.111987): T lives on 0..7, so F_avg/F_HS are code-isometry diagnostics. Code-isometry converged=True.

W=96 has unit norm (p_lost 7.009e-13) but 64 vs 96 full-column overlap is 0.980 because the pad tail (levels ≥64) is still moving. Unit norm ≠ full-space convergence. full_space_pad_tail_converged_64_vs_96=False.

## Conformance tests (W=96 unless noted)

- D(0)=I: err 1.14e-14. SNAP unimodular max err 0.00e+00.
- D(−α)D(α) on |0⟩,|3⟩,|7⟩ norms [1.0, 1.0000000000000002, 0.9999999999999999]; on |+⟩ 1.000000. Full-space D†D−I fro/n 3.400e-02 (residual on high-n basis states near the cutoff).
- Sequence then inverse, back to code basis: fro/col 7.501e-07.
- Same tests at W=32 inverse_ok=False roundtrip_code_ok=True (32 is too small for three |α|≈1.8 displacements).
- Cropped expm(α(a†−a)) on W vs +32-restricted true D, fro/n 3.352e-02.

propagation_tests_pass=True. New conformance evals 52 (not SK, not added to 2048).

## Claim

No physical validation of compiled W_BinH. WCOMP_FAIL_BUDGET stays the historical SPSA result under a broken inter-gate projection. Corrected frozen-pulse numbers above are diagnostics of that same pulse, not a new compile. No GR.

# Abort audit and local finite-difference check

The original pilot files under `results/FT/cells`, `results/FT/train`, `SUMMARY.md`, and `FINDINGS.md` are unchanged. This directory is the separate archive. Validation and final seeds were not read. No N=12 or N=15 sweep was run.

## What the 1e-9 abort was

166 restricted-displacement starts aborted. Query counts on replay of the same seeds match the original ledger in 166/166 cases, and the scored energy matches the start's initial energy. There are no ledger mismatches. QAOA, Nov3rd, and the unitary strict-logical D8 diagnostic had zero aborts. The gate hits only the contractive restriction of the true displacement.

Every captured abort state has NF=16 norm deficit between `1.00e-9` and `7.2e-9` (median `1.52e-9`). An independent oscillator at NF=48 is trace-preserving there: norm deficit about `1e-15`, and population on levels ≥ 32 is about `1e-30`. The displacement slice from a 32-level pad and a 64-level pad agrees to about `1e-15`, and the unitary exponential of the cropped 16-level generator is unitary to about `1e-16`. This is not an eigendecomposition or Padé error.

The NF=16 norm loss is the physical contraction of that restriction. It does not always equal the final population above 16 in the NF=48 state, because amplitude that leaves and re-enters is deleted by the restriction. The observable effect is still small: fidelity changes by at most `3e-11`, and energy changes by `1.5e-9` to `5.5e-8` (median `9e-9`; 41% of the abort states exceed the `1e-8` cutoff budget). NF=32 matches NF=48 to about `1e-14` in energy, fidelity, leakage, and norm.

So the abort is an erroneous unitarity test on a truncation the cutoff budget already treats as small, and it penalizes Nov9 only. It is not a license to renormalize, and it is not a reason to raise the `1e-9` gate. The trace-preserving fix is to keep the gate and put the evolution in a cutoff that actually retains the tail.

## Replay

Only the 145 dev records that contained an abort were re-run, at NF=32, with the original seeds, budgets, and `1e-9` gate. None aborted. Training vectors for the three affected L=3 runs changed by at most `2e-9`, so their held-out finetunes were not re-initialized. Per-record original and replay metrics are in `results/FT/replay_nf32/CHANGES.json`.

Where the aborted start was not the selected start, reported F and E move by far less than `0.01`. Two cold selections do move, because every usable start had been replaced by its initial point:

- KM K-prior, N=6, b=1, L=3, seed 10004, cap 1024: F 0.614 → 0.786, E -5.23 → -5.81. Still above the product energy `-6`.
- U0, N=9, b=1, L=3, seed 10004, cap 1024: F 0.021 → 0.647, E -2.14 → -8.93. The completed run returns to the product state. The original number was the initial point of a start that the gate had killed.

The 65 aborted finetunes had been scored as their transferred initial point, which was near the product state. Letting SPSA finish usually leaves that point and lowers fidelity, sometimes from about 0.97 to about 0.01. That is the finite-budget SPSA outcome, not evidence that a different initialization or a longer budget cannot do better.

## Finite differences, not an optimizer

Central differences of the exact energy around the deterministic cold initialization (two seeds, both N, both b, L=1, 2, 3; steps `1e-4` through `0.05`, plus three joint Gaussian probes) are in `finite_difference.json`.

The initialization is not flat. Slopes in γ and α are typically several units, and the curvature in those directions is positive. Most one-coordinate steps lower the energy of the echo initialization, and the downhill direction is back toward the product state. None of the probed steps beat the product energy `-bN`. The product state itself sits above the ground state (the gap at N=9, b=1, seed 10000 is about 1.15). This is a local statement about this initialization under exact energy. It is not a proof that the ansatz cannot represent a better state.

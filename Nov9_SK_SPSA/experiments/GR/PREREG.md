# GR preregistration for the bounded pilot

Parent protocol commit `92e094dd`. Numerical choices live in `prereg.json`. D2–D5 were approved by Zach at 18:41 Asia/Shanghai on 2026-11-09 as simulation assumptions, not as a hardware model. Validation and final stay closed. Fidelity is unconditioned and is not a selector.

## Reconciliations with the consultation text

- The 4-start budget in `PROTOCOL.md` §5 is the one used: 8 calibration queries, 123 iterations, and 2 candidate queries per start at cap 1024. The older “two starts” sentence in Idea 2 is not used. The best of the 4 evaluated candidate energies is kept.
- Caps 128 and 1024 are separate runs. A prefix of the long trajectory is not the short-cap estimator, because the last-iterate and Polyak choice happens at the cap. For cap 128 the same rule gives 11 iterations per start.
- In-budget monitoring is 0. Initial energies and fidelities are oracle diagnostics and are not charged against the cap.
- One SPSA configuration is used for every method: `c` from the leak probe (start 0.1, halved at most twice if a `c σ_α` step changes leak probability by more than 0.01), target median initial step `0.5σ`, clip `2σ`. No per-method search on these 8 seeds.
- The α step is `c σ_α` in α, mapped through `u` by the derivative of `0.30 tanh(u)`. The coordinate freezes if `sech²(u) < 1e-4`.
- Polyak is the mean of the last `min(64, n_iter)` post-update vectors.
- Uz encoding arms share `method_id` 2, so their SPSA noise matches. Gray uses id 5, U0 uses 3, strict-logical D8 uses 4, Nov3rd uses 1, QAOA uses 0, as in `protocol_draft.json`.
- Random encodings use `SeedSequence([seed, N, 1000+r, 2])` and do not depend on `b` or `L`. They read no `J`.
- QAOA’s deterministic start is `γ_ℓ = β_ℓ = 0.25`. Its random box is `U(-0.5, 0.5)` on each angle. Nov9th start E is `γ = 0.25`, `θ = 0`, `α_ℓ = (-1)^{ℓ+1} 0.06` with `ℓ` starting at 1, `s = 0`.
- Nov3rd uses that same `γ`, `θ`, and `s` on the interleaved packing, with ideal `W`, and no displacement.
- Greedy grouping is sequential over remaining triples, with exact-float ties broken by the increasing site triple. Its pair-read count is `O(N^4)` for this N, which is recorded, rather than the consultation’s `O(N^3)` label. It never sees a ground state.
- anti-R reverses the R triple `[b0, b1, b2] → [b2, b1, b0]`.
- Production true `D` is the unitary exponential of the oscillator generator truncated at `NF_sim`. It is not an 8-cropped exponential and states are not renormalized. `NF_sim` is the smallest ladder value whose stress-grid gap to NF = 48 is at most `1e-8`, then checked again on the selected points.
- Strict-logical `D8` is a diagnostic. It is not called a displacement.
- A failed run scores the deterministic start and reports the queries actually used. Completed runs assert the counter equals the cap.

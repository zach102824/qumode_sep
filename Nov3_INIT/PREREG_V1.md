# N3INIT v1 preregistration

Development stream only. This is not the frozen 1000-seed benchmark, not a retune of recipe `N3X_matched_v1`, and not a Nov9 or SYK experiment. GR owns that benchmark. LE owns diagonal conditioning. Local owns SPSA gain schedules and the charged L1→L3 transfer. This file freezes the v1 comparison before any SPSA energy is computed.

## Circuit (unchanged original Nov3)

Logical Hamiltonian, entropy field only:

\[
H = \sum_{i<j} J_{ij} Z_i Z_j - B^\star \sum_i X_i.
\]

`J` is `default_rng(seed).normal(0, 1/sqrt(N))` on the upper triangle, symmetric, zero diagonal. The same `J` and the same `B*` are used for Nov3 and QAOA.

Original Nov3, full three shell parameters per layer. One `γ`, one `θ_K`, and one `θ_M`, identical in every layer. No per-layer gamma and no extra gate.

\[
|+\rangle \to \big[U_{cZ}(\gamma,\theta_K,\theta_M) \to WSW(s_{\ell,1},s_{\ell,2},s_{\ell,3})\big]^L.
\]

Packing is the interleaved triple `[[a, a+m, a+2m]]` with `m = N/3`. The shell phase on a cavity is `s1 * p + s2 * C(p,2) + s3 * [p=3]`, `p` the cavity popcount. `K = sym(offdiag(J @ J))` and `M` the inter-cavity mean of `J` (intra-cavity entries zero), each RMS-matched to upper-triangle `J`. The cost diagonal is `γ (J + θ_K K̂ + θ_M M̂)`. The physical energy uses `J` and `B*` only. Features are not added to `H`.

QAOA at `p = L` is the matched baseline, not a reduced Nov3:

\[
|+\rangle \to \big[e^{-i \gamma_\ell H_{ZZ}} e^{-i \beta_\ell \sum X}\big]^L,
\]

with independent angles. `H_ZZ` is the `J` diagonal only.

Algebraic identity used only as a conformance check and as the center of the proposed start distribution: if `θ_K = θ_M = 0`, `s2 = s3 = 0`, and `s1_ℓ = -2 β_ℓ`, Nov3 with a shared `γ` matches QAOA with every `γ_ℓ = γ` up to a global phase (fidelity 1, same energy). The circuit is not rewritten into that restriction during optimization. All three shell parameters stay free.

## B*

Classical, charged as classical time, never as a quantum query, and never as a start selector.

- Cut A = sites `0 .. floor(N/2)-1` in LSB order (site `i` is bit `i`). Odd `N` leaves the two sides different by one spin. The cut is not optimized.
- Entropy is the natural log of the squared Schmidt coefficients across that cut, singular values squared, coefficients below `1e-300` dropped.
- Grid: 60 points on `[0.05, 3.0]`. Peak must be interior. Then `minimize_scalar` bounded on the neighboring brackets, `xatol = 1e-6`.
- Ground state from a real symmetric sparse solve. Global sign so the amplitude sum is positive. Require gap `E1-E0 > 1e-8`, even parity `> 1-1e-6`, numerical positivity `min > -1e-8`, residual `< 1e-8`. Record the even-sector and odd-sector gaps. Fidelity and `F_+` are stored after selection. They do not rank starts.

Conformance reproduces the published `N=9`, seed `10000` value `B* = 0.5672817157081989` within `1e-5`. That seed is not in this optimization panel.

## Seeds and split

Optimization seeds are only `10018..10025`.

- Train half, the only half that may choose an arm: `10018..10021`.
- Test half: `10022..10025`. The test half is not read before `FREEZE.json` is written, and it does not change the freeze.

Validation `11000..11031` and final `20000..20999` are refused. Seeds outside `10018..10025` are refused by the runner. GR and local dev seeds are not reused.

## Budgets

Primary grid, and the only grid in v1: `N ∈ {9,12}`, `L ∈ {1,3}`, `q ∈ {128,512,1024}`.

Inclusive quantum budget `Q = q * L`. Every screen, init, calibration, iteration, restart, and selection energy is inside `Q`. Monitoring queries are zero. A completed cell has ledger total exactly `Q`. `N=15` and `N=18`, and layers `2,4,5,6`, are not in v1. They wait on a later preregistration after this grid finishes with exact ledgers.

`q = 1024` here means `Q = 1024 * L` for that one cell. It is not a label pasted onto a larger uncapped search.

## Shared SPSA block

SPSA only. No BFGS and no natural gradient. Gains are the frozen matched-recipe gains, not retuned on these seeds:

- perturbation `c = 0.1`
- `A = 25`, exponents `0.602` and `0.101`
- four two-sided calibration pairs
- `a0 = 0.5 / median(|g_j|/σ_j)`, or `0` if that median is `< 1e-15`
- step clipped at `2σ`
- Rademacher `±1`
- σ: Nov3 `γ 0.10`, `θ 0.25`, shells `0.15`; QAOA angles `0.10`

Selectable points inside a block, energy only: the charged init, the last iterate, the mean of the last `min(64, T)` iterates, and the midpoint iterate. Lowest energy wins. An energy tie keeps the init. Across starts, lowest energy wins and an energy tie keeps the smaller start index.

A block with `T` iterations costs `1 init + 8 calibration + 2T iteration + 3 selection = 12 + 2T`.

RNG is `SeedSequence([seed, N, round(10*B*), L, method_id, start_id, stream])`. Stream `2` draws the start. Stream `0` drives SPSA. `round(10*B*)` is the frozen recipe's key; it can alias fields that share a tenth, and it is not changed here.

## Arm equal4 (baseline)

Recipe name `equal4`. Method ids: QAOA `0`, Nov3 `1`. Same four-start split as `N3X_matched_v1`, scaled only by the inclusive `Q` above. Gains, boxes, and the deterministic center are not retuned.

- Start 0: Nov3 `γ=0.25`, `θ=0`, shells `0`. QAOA every angle `0.25`.
- Starts 1–3: Nov3 `γ,θ ~ U(-0.5,0.5)`, shells `~ U(-0.3,0.3)`. QAOA angles `~ U(-0.5,0.5)`.
- `T = Q/8 - 6`. This is an integer on every v1 budget (`10, 58, 122` at `L=1` and `42, 186, 378` at `L=3`).
- Four blocks fill `Q` exactly. Both methods get this allocation.

## Arm halve8to2 (proposed)

Recipe name `halve8to2`. Method ids: Nov3 `70`, QAOA `71`. Same SPSA block and the same σ. The only changes are the cold-start distribution and how `Q` is split. QAOA receives the same split and the same correlation structure.

Instance-independent correlated draws. They do not read `J`, the ground state, or fidelity.

- Start 0, deterministic center. Nov3: `γ=0.25`, `θ_K=θ_M=0`, every layer `(s1,s2,s3)=(-0.5,0,0)`. That center is the algebraic QAOA point `β=0.25`, not a fitted state. QAOA start 0 stays every angle `0.25`.
- Starts 1–7. Draw one shared cost vector and one shared mixer, then a layer residual so the layers are not independent:
  - Nov3: `γ,θ_K,θ_M ~ U(-0.5,0.5)`, base shell `~ U(-0.5,0.5)^3`, each layer adds `U(-0.15,0.15)^3`.
  - QAOA: shared `(γ,β) ~ U(-0.5,0.5)^2`, each layer adds `U(-0.15,0.15)^2`.
  - Realized coordinates therefore lie in `[-0.65,0.65]`. There is no `π` box and no second-stage widening.

Allocation inside `Q`:

1. Charge one energy at each of the 8 starts (`screen`).
2. Keep the 2 lowest energies. Tie-break: smaller start index. Fidelity is not consulted.
3. From each kept `x0`, run a fresh SPSA block. Its init energy is charged again. `T = (Q - 32) / 4`, an integer on every v1 budget (`24, 120, 248` at `L=1` and `88, 376, 760` at `L=3`).

Ledger: `8 + 2*(12+2T) = Q`. Discarded starts are paid for and are not free in a later arm.

This is not an L1→L3 transfer. Depth is not optimized at `L=1` and embedded.

## Selection, fidelity, failures

Start ranking and the SPSA candidate ranking use energy only. Fidelity, `η = (F-F_+)/(1-F_+)`, and escape `E < E_+ - 1e-3 N` with `E_+ = -B* N` are computed after the winning state is chosen.

A non-finite energy fails that cell. The cell is kept as a failure and is not replaced by the product-state energy. Freeze refuses to adopt anything if any train cell is missing or failed.

## Freeze rule (train half only)

For each `q` separately, over the 16 train points (4 seeds × `N∈{9,12}` × `L∈{1,3}`):

- `d_self = E(Nov3, halve) - E(Nov3, equal4)`
- `d_gap(arm) = E(Nov3, arm) - E(QAOA, arm)`

Adopt `halve8to2` at that `q` only if `mean(d_self) < 0` and `mean(d_gap(halve)) ≤ mean(d_gap(equal4))`. Otherwise the frozen recommendation stays `equal4`. The test half is then run for both arms and both methods. It does not edit the freeze. The next adaptation, if any, needs its own preregistration and may not treat this test half as unseen.

## Reporting

For each `(split, arm, N, L, q)` report `n`, failures, mean and median `E` and `F`, tails (min and max), paired Nov3−QAOA `ΔE` and `ΔF`, energy wins, fidelity wins (post hoc), and a seed-cluster bootstrap of the mean: 4000 resamples, `default_rng(3334)`, percentiles 2.5 and 97.5, seeds resampled with `N` and `L` held in the cell. Query totals and wall time are reported. Both methods are reported. A Nov3 win is not the only summary.

Bootstrap call order is fixed: splits `train` then `test`; `q` ascending; `N` ascending; `L` ascending; within a cell the statistics are mean `E` Nov3, mean `E` QAOA, mean `F` Nov3, mean `F` QAOA, mean `ΔE`, mean `ΔF`.

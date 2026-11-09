# N3INIT v5 preregistration

Exact v4 four-start recipes, frozen before any N=18 instance or SPSA. `center4` remains a development candidate. It is not a GR benchmark adoption. Prior cells are not overwritten and are not relabeled. Historical research queries `1679360` are spent once.

## Recipe freeze (exact v4)

Original Nov3, all `Uc` controls shared across layers, full 3-shell parameters per layer, unchanged circuit. QAOA `p = L`. SPSA only, frozen gains: `c = 0.1`, `A = 25`, exponents `0.602/0.101`, 4 two-sided calibration pairs, `a0 = 0.5 / median(|g|/σ)` or 0 if that median `< 1e-15`, clip `2σ`, Rademacher `±1`. σ: Nov3 `γ 0.10`, `θ 0.25`, shells `0.15`; QAOA angles `0.10`. Four equal blocks, `T = Q/8 − 6`. Inclusive `Q = q L`. Energy selection among init, last iterate, Polyak of last `min(64, T)`, midpoint. Lowest energy; energy tie keeps init; across starts lower `start_id` wins. Fidelity is post hoc. No GS/fidelity init or start selection. No architecture change, no extra parameters, no conditioning, no frame combo, no BFGS, no natural gradient.

Nov3 start 0: `γ = 0.25`, `θ_K = θ_M = 0`, every layer `(s1, s2, s3) = (−0.5, 0, 0)`. QAOA start 0: every layer `(γ, β) = (0.25, 0.25)`, the derived image of that center, already overlap-checked at N=6 L=1. Starts 1–3 stay in the historical boxes. Method ids remain 1 (Nov3) and 0 (QAOA). No new adaptation after seeing N=18 outcomes.

## Arms

New directory `results/v5/`.

- `nov3_center4`
- `qaoa_center4`

`nov3_equal4` is not rerun. The N=15 L1 self-difference was a nonnegative tie on almost every seed; the hypothesis here is whether the Nov3–QAOA energy gap persists at N=18, not whether start 0 is selected.

## Grid

Seeds `10018..10025` only, n=8 development instances, not a population or 1000-seed claim. N=18, L=1, `q ∈ {512, 1024}`, `Q = 512` and `Q = 1024`. Both budgets run. Energy-only selection. Fidelity post hoc. Report mean `(E − E0)/N` and F, paired mean CIs, medians, wins, tails, actual n, failures.

N=18 at this size is new. Provenance check: refuse any seed in `11000..11031` or `20000..20999`; refuse any seed outside `10018..10025`; refuse to open an instance file if a prior result for that `(N, seed)` already exists under this stream. This band was used at smaller N. It is development, not a holdout.

`B*` is the interior entropy maximum on the balanced cut `0 .. floor(N/2)−1` = sites `0..8`. Odd-N vs even-N cut size is the existing rule. Contract: interior grid argmax, `gap > 1e-8`, `gs_parity > 1 − 1e-6`, `gs_min > −1e-8`, residual `< 1e-8`, both sector gaps positive, even GS. Classical prep is not a quantum query and not a selector. Record its wall, CPU, and peak RSS separately.

## Resource gate (before instances)

Smoke on seed `10018` only, no SPSA, no other seeds:

1. One sparse `eigsh` at `b = 0.5` (statevector `2^18`, no dense Hamiltonian).
2. One original-Nov3 L1 energy at the frozen center, dummy `| + ⟩` for the unused GS slot.
3. RSS of that process, including the 2 MB-scale state and the `hc` diagonal.

Gate `n18_allowed` if all of: eigsh `< 30 s`, L1 energy `< 0.5 s`, peak RSS `< 4 GiB`, `dim = 2^18`, `dense_matrix = false`. If the gate fails, stop, write the probe, and do not open the eight instances. Do not switch to a cheaper seed. Do not drop to a favorable subset. No L=2..6 grid.

If the gate passes, build the eight `B*` files with 4 workers, BLAS/OMP 1, checkpoint each instance to disk before the next SPSA cell, then run the 16 cells (2 methods × 8 seeds × 1 of each q, actually 32 cells: 2 arms × 8 seeds × 2 q). Resume by skipping existing cell files.

## Read rule

Energy only. No fidelity win is required.

- `d_pair = E(nov3_center4) − E(qaoa_center4)`
- `persists` if `mean(d_pair) < 0` and the seed-bootstrap 95% interval excludes 0 at both q=512 and q=1024.
- If the interval contains 0, report the point estimate; do not call persistence confirmed.
- If `mean(d_pair) > 0` and the interval excludes 0, the N=15 shallow energy edge does not persist at N=18 on this band.

Bootstrap: 4000 resamples, `default_rng(3334)`, seeds resampled. Call order: q ascending. Both budgets are reported. Outcomes do not change Q, gains, or the center.

## Queries

Each cell ledger equals its own `Q`. Report the procedural cap, the sum of new v5 quantum queries, and the cumulative research total `1679360 + new`. Classical `B*` wall/CPU/RSS is reported separately and is not added to the quantum ledger. No uncharged training. Prefixes already fitted stay ledgered.

## Off limits

GR frozen 1000 panel. Validation `11000..11031`. Final `20000..20999`. LE kernel throughput `10032..47`. Local R8 N15 frame `10026..31`. Nov9, SYK, AF, N21, extra agents, extra parameters, conditioning, frame combo.

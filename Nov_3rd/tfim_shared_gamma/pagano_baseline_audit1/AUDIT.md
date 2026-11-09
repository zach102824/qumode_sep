# Pagano baseline audit 1

One bounded audit of whether the recorded N=12, p=12 QAOA fidelity
`F=0.514718536289` should be judged against Pagano et al., arXiv:1906.02700.
No new target was optimized. The frozen manifest is unchanged.

## Sources read

- Main text and the HTML rendering of the supplement: https://arxiv.org/abs/1906.02700 and https://arxiv.org/html/1906.02700
- No separate code repository is cited in the paper. This checkout has no archived Pagano-discrepancy note beyond the depth-2 seed offsets named `pagano_` in the existing optimizers.
- Figure S3 states an empirical scaling. It does not print a number at N=12, p=12. None is invented here.

## What the paper defines

Equation (1):

`H = sum_{i<j} J_ij σ^x_i σ^x_j + B sum_i σ^y_i`, with `J_ij > 0`.

`J_0` is the average nearest-neighbor coupling. Experimental `J_ij` come from the ion normal modes (supplement Eq. 20). For N<20 they are approximated by `J_0 / r^α`. Figure S5 shows that a pure power law misses the long distances, and quotes a compound fit for the N=12 gradient-descent data, `{J_0, α', β'} = {0.580, 0.322, 0.229}`.

Equation (2), with the product applied so that layer k=1 acts first, as Figure 1 draws `H_A` then `H_B`:

`|β,γ⟩ = prod_{k=1}^{p} exp(-i β_k H_B / J_0) exp(-i γ_k H_A / J_0) |ψ_0⟩`.

`|ψ_0⟩` is the ground state of `H_B` for `B<0`, the product state `|↑⟩_y^{⊗N}` with `|↑⟩_y = (|↑⟩_z + i|↓⟩_z)/√2`.

Equation (3):

`η = (E(β,γ) − E_max) / (E_gs − E_max)`.

This is a rescaled energy. It is not `|⟨GS|ψ⟩|^2`. The supplement says the generic mixer `H_B = −sum σ^y` and then, in Figure S3, plots η and the squared ground-state overlap as two different quantities. The overlap panel is described as scaling roughly as `p/N` for ideal `α=1.1`, for `p = 0…5` and `N = 8…20`.

Numbers that are printed, and the setting they belong to:

| Where | N | p | Quantity | Value | Setting |
|---|---|---|---|---|---|
| Fig. 2a | 20 | 1 | η experimental / η theory | (93.8±0.4)% / 96.1% | `J_0/2π=0.57 kHz`, `B/J_0∼−0.3` |
| Fig. 3a | 12 | 1 | η experimental | (94.9±0.2)% | `J_0/2π=0.57 kHz`, `B/J_0=−0.3` |
| Fig. 3b | 20 | 1 | η experimental | (94.7±0.1)% | `J_0/2π=0.55 kHz`, `B/J_0=−0.3` |
| Fig. 2d text | 20 | 2 | η experimental | (93.9±0.3)% | same apparatus as the p=1 system-2 scan |
| Fig. 4a | 12 | 1 | angles, not a fidelity | `β*=0.25`, `γ*=0.31` | `B=0`, `J_0/2π=0.33 kHz`, sampling of `H_A` |
| Fig. S3 | 8–20 | 0–5 | η and squared overlap | no cell printed | ideal `α=1.1`, `B` at the finite-size entropy maximum |

There is no published squared fidelity at N=12, p=12.

## Rotation from XX+Y to ZZ+X

The orthogonal map on the Bloch vector

`(X, Y, Z) → (Z, X, Y)`

sends `σ^x_i σ^x_j` to `σ^z_i σ^z_j` and `σ^y_i` to `σ^x_i`. It preserves the right-handed Pauli algebra. Because every `J_ij` is positive, the image is antiferromagnetic ZZ, not a ferromagnet and not a GHZ cat. A negative `B` stays a negative coefficient of `sum X`. The initial `|↑⟩_y` maps to the real product `|+⟩`, which is the ground state of `B sum X` when `B<0`.

After that map, Figure 1's order is `exp(−i γ H_ZZ) ` then `exp(−i β_paper (B/J_0) sum X)`. Our simulator applies `exp(−i γ H_c)` then `exp(−i β sum X)`, with `H_c = sum_{i<j} J_ij Z_i Z_j`. The generators match. The angle units do not: `β_ours = β_paper (B/J_0)`. With our `J_0 = 1` and `B/J_0 = −0.3444`, a published β is not a number that can be pasted into our parameter vector. No such paste was done.

Flipping the sign of every angle on the stored p=12 vector produces the complex conjugate of the stored state (overlap 1, energy residual 0). The stored gammas are all negative and the betas all positive, so this state is the conjugate of a positive-gamma, negative-beta schedule. Energy and full-space squared fidelity are unchanged. The supplement's "both angles non-negative" statement is about their own angle convention and their own minima. It is not a failure of this vector.

## Our frozen target

`H = sum_{i<j} |i−j|^{−1.1} Z_i Z_j − 0.3444 sum_i X_i`, open chain, N=12, manifest SHA256 `14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9`.

Every nearest-neighbor entry is 1, so `J_0 = 1` and `B/J_0 = −0.3444`. The ground energy recomputed here is `−8.613202458935264`, against the contract `−8.6132024589353`. The highest eigenvalue is `23.79999934975569`.

This is the ideal power-law family of Figure S3, not the experimental ion matrix of Figure S5, and not their entropy-maximizing field at each N. The paper locates `|B/J_0| = 0.31` for 20 ions and a half-chain entropy peak near `B/J_0 ∼ −0.33`. Our field is the frozen `−0.3444`. It was not moved.

## Implementation checks

A dense `expm` reference on the N=6 frozen target, three random layers, matches `Runner` to `1.18e−16` in the state and `6.4e−16` in the energy. Central differences of step `1e−6` match the analytic gradient to `2.5e−10`. The stored p=12 gammas are not tied: peak-to-peak spread is `0.307`. Replaying `x_exact` returns

- `E = −8.325407549783135`
- `F = |⟨GS|ψ⟩|^2 = 0.5147185362888936` (residual `−2.2e−16` against the published file)
- `Var(H) = 0.42114525455677665`
- same-target post-hoc `η = 0.9911210589175745`
- state norm `1 − 2e−16`

`η` is high because the energy error `0.288` is a small fraction of the bandwidth `E_max − E_0 ≈ 32.41`, while the full gap is about `1.6e−3`. Squared overlap and η answer different questions. The repeated local minimum is not certified as the global minimum. The checks do show that it is not an artifact of a tied gamma or a broken gradient.

The supplement bootstrap for `q=2` uses `β = (β*, β*−0.2)` and `γ = (γ*, γ*+0.2)`, then linear interpolation at `q=3` and cubic interpolation for `q>3`, with knots `s_i = (i−1)/(p−1)`. Our depth-2 seeds use that ±0.2 offset. Later rescues resample on layer centers `(l+0.5)/L`. That is a different initializer, not a different unitary. The remembered start `(β, γ) = (0.01, −0.01)` does not appear in the sections read and is not used as a paper protocol.

## Matched status

Incomparable. The requirement "QAOA fidelity should be at least as good as Pagano" mixes `|⟨GS|ψ⟩|^2 = 0.515` on this frozen α=1.1, `B=−0.3444`, p=12 state with printed η values near 0.94–0.96 at p=1 or p=2 on experimental couplings. Our same-target η of 0.991 is a post-hoc label on the already selected energy-minimizing vector. It is not a reproduction of a paper cell, and it was not used to train or to select the state.

No circuit bug was found, so the stored QAOA results are not superseded. No alternative Hamiltonian was optimized.

## Effect on the L5/L6 comparisons

Those comparisons are internal: same frozen target, same energy-only selection, same full-space squared fidelity. This audit does not move their energies or fidelities. It also does not turn them into a claim about Pagano's η. The restricted N=12 win stays provisional as a comparison against this QAOA baseline, because a local optimizer is not a global optimum. It is not provisional because of a missing factor that would raise `F` to a paper η.

## Proposed reproduction contract

Not run. A later slice that wants a paper-matched number has to state, before any optimization:

1. Ideal Figure S3: `J_ij = |i−j|^{−1.1}`, `J_0 = 1`, `B` equal to the half-chain-entropy maximizer at that N (or an explicit fixed `B/J_0`), open chain, both η and `|⟨GS|ψ⟩|^2`, `p ≤ 5` so the comparison stays inside the plotted range.
2. Or experimental Figure 3a: the N=12 ion matrix from supplement Eq. 20, `B/J_0 = −0.3`, p=1, observable η, with `J_0` the average nearest neighbor in the same units as `B`.

Either contract is a different target from the frozen manifest. It needs to be shown before any new optimization.

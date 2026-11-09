# Angle conversion and seed rules

Pinned parent audit: commit `95a6b3590adf561e0ef6433f548b2ed23f8bc55a`.
Paper: Pagano et al., arXiv:1906.02700, supplement section I.1.2, as rendered at
https://arxiv.org/html/1906.02700. No analytic p=1 formula is printed in that
section. This run does not invent one.

## What is the same in both arms

The circuit is the existing simulator in
`Nov_3rd/tfim_shared_gamma/round5_KS/model.py` (`Runner.forward` for `qaoa`):
`|+⟩`, then each layer applies `exp(-i γ H_c)` and then `exp(-i β sum X)`.
`H_c` is the frozen ZZ matrix. Gamma and beta are independent at every layer.
Training and selection use energy only. Fidelity, η, and variance are computed
on the energy-selected vector.

The optimizer is L-BFGS-B with analytic gradients, `maxiter=1500`, `ftol=1e-14`,
`gtol=1e-8`, and `maxfun=600`. The 600-call ceiling is this slice's cap. Slice 3
used 500. Both arms here use 600, so that change is not an endpoint effect.

## Paper units

Equation (2) evolves with `β (H_B / J_0)` and `γ (H_A / J_0)`.
Equation (1) is `H_A = sum_{i<j} J_ij σ^x_i σ^x_j` and `H_B = B sum_i σ^y_i`.
The Bloch map `(X, Y, Z) → (Z, X, Y)` sends this to ZZ + B X. With the frozen
nearest-neighbor couplings equal to 1, `J_0 = 1`, so

- `γ_ours = γ_paper`
- `β_ours = (B / J_0) β_paper`

`B` is the manifest string `"-0.3444"`. `B/J_0 = -0.3444`. This factor is used
only on a paper angle. It is not applied again to an angle that is already in
our vector.

## p = 1

Supplement step 1: "For q=1, use an analytic formula to find β*(1) and γ*(1)."
The formula is not in the HTML section and it is not in this repository. Both
arms therefore share three numerical starts drawn from `uniform(-0.8, 0.8)` on
gamma and on beta. That box is the fresh QAOA box in
`round5_KS/optimize.py`, `fresh_vector`. It is not a paper rule. The remembered
start `(β, γ) = (0.01, -0.01)` is not used.

## p = 2, shared, mapped

Supplement step 2: `β^(2) = (β*(1), β*(1) - 0.2)` and
`γ^(2) = (γ*(1), γ*(1) + 0.2)`. The 0.2 is in paper units.

Mapped onto the energy-selected p=1 angles `(γ, β)` already in our units:

- gamma becomes `[γ, γ + 0.2]`
- beta becomes `[β, β + (-0.3444)*(-0.2)] = [β, β + 0.06888]`

The historical code does not do this. `round5_KS/optimize.py` builds
`[g0, g0+off]` and `[b0, b0-off]` with `off` in `(0.2, 0.0, 0.1)`, which is the
paper's ±0.2 written directly in our beta. The audit's statement that this
"matches" was a match in unconverted paper units. This slice uses the mapped
offset, once, for both arms. The two extra starts are `N(0, 0.05²)` on that
vector. The 0.05 width is ours, not the paper's. This whole p=2 rule is a
shared factor. A difference between arms is not attributed to it.

## p ≥ 3, the only arm difference

Supplement steps 5–6, for a step into depth `q` from `q-1` angles:

- source abscissa `s_i = (i-1)/(q-2)`
- linear interpolation when `q = 3`, cubic when `q > 3`
- sample at spacing `Δs = 1/(q-1)`, which is `s = 0, 1/(q-1), …, 1`

The attached review said to interpolate linearly at every depth. That is not
what the supplement says. Both arms use linear only when the source has two
angles (the step into p=3) and a natural cubic thereafter. The cubic is
`scipy.interpolate.CubicSpline(..., bc_type="natural")`, the same call the
center code already makes.

Center knots are the existing ones, `(l + 0.5)/L`, from
`round5_KS/optimize.py` `resample_rows` and from
`round6_QR/slice3/run_slice3.py` `_resample`. Values requested outside the
source knots are clipped to the end knots, which is what those functions do
(`np.interp` ends and `np.clip` on the spline).

Endpoint knots use the supplement abscissae. The same clip is applied. On that
grid the sample points lie inside the source interval, so the clip does nothing
and the endpoints of the schedule are copied exactly.

With two source angles, the clipped center interpolant and the endpoint
interpolant return the same three numbers. The p=3 starts therefore coincide.
They are executed once and charged once. The first depth at which the seeds
differ is p=4. From there each arm resamples its own energy-selected schedule.
Matched Gaussian noise (`σ = 0.05`, seed without an arm id) is added to each
arm's own base, so the noise is not a second difference.

No stored-p12 neighborhood is run.

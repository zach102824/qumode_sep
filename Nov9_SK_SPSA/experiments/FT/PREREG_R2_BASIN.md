# PREREG R2 B2 BASIN (written before any SPSA at B*)

Source: `NEXT_IDEAS_R2.md` at `82dd41a5` (PR 75), idea B2. User scope: entropy B* only. Imaginary-axis displacement (idea B1) is not run.

This file is the recipe. The b=1 and b=1.5 pilot is a reference and is not repeated.

## Instance

- J: `default_rng(seed)`, normal, standard deviation `1/sqrt(N)`, upper triangle row-major.
- Cut, fixed: sites `0 .. floor(N/2)-1` versus the rest. LSB = site 0. Not chosen per instance.
- Entropy: natural log (nats), matching the LE diagnostic. Bits = nats / ln(2). The argmax does not depend on that factor.
- B*: 60-point grid on `[0.05, 3.0]`, then `scipy.optimize.minimize_scalar` (bounded, `xatol=1e-6`) on the neighboring bracket, maximizing `S_A`. A peak on a grid edge invalidates the instance.
- Stability, reported and not used to pick B*: the same maximizer on a 90-point grid.
- Checks, required: b>0, GS even, strictly positive after the global-sign fix, `E1-E0 > 1e-8`. Also store `Δ_even` and the parity of E1. B* enters the circuit only as the coefficient of `-sum X`. It is not a feature and not an initial state.
- Dev inference seeds `10000..10007`, N=6 and N=9. Validation `11000..11031` and final `20000..20999` are not read.
- One σ calibration instance, not reported as a result: N=9 seed `10008`.

## Circuit and cutoff

- Nov9: `|+> -> [U_cZ -> D(α_ℓ) -> bare shell SNAP_ℓ]^L`. Real α, `|α|≤0.30`, shared γ, shared K and M, three shell phases per layer. True displacement, matrix exponential in `NF+32` then restricted. No renormalization.
- NF=32 for every Nov9 energy. Gate: norm deficit `> 1e-9` aborts that start and scores the initial point.
- Before SPSA, one probe at seed 10008, N=9, L=3, all `α=0.3`: population above 32 at NF=48 and `|ΔE|` versus NF=32 must be `≤ 1e-8`, or the run stops.
- After SPSA, the same comparison on every selected cap-1024 Nov9 point. Failures are flagged, not dropped.
- U0 (whole component unphased if any cavity leaks) is a sensitivity at cap 1024 for the cold and wide arms. Uz is primary.
- QAOA: `p=L`, independent angles, same SPSA budget. No imaginary displacement.

## Arms and inits (same parameter count)

Four starts. Stream 2 draws the init, stream 0 draws SPSA, `SeedSequence([seed, N, round(10*B*), L, method_id, start_id, stream])`. One B* per `(N, seed)`, so the rounded field does not alias two instances.

- `basin_cold_uz` / `basin_cold_u0`: start 0 is γ=0.25, θ=0, `α_ℓ=(-1)^{ℓ+1} 0.06`, s=0. Starts 1–3 are γ,θ ~ U(-0.5,0.5), α ~ U(-0.08,0.08), s ~ U(-0.3,0.3).
- `basin_wide_uz` / `basin_wide_u0`: start 0 is the real-α-plus-phase frame, γ=0.25, θ=0, the same echo α, `s1` of the first layer `+π/2` and `s1` of the last layer `-π/2` (one layer: `-π/2`). Starts 1–3 are γ,θ ~ U(-1,1), α ~ U(-0.30,0.30), s ~ U(-π,π). No α outside 0.30.
- `basin_qaoa`: start 0 is every angle 0.25. Starts 1–3 are every angle ~ U(-0.5,0.5).

## Budget

Caps 128 and 1024 for cold, wide, and QAOA. U0 only at 1024. Every circuit energy call is inside the cap, including the evaluation of each start's `x0`, calibration, iterations, and both candidate evaluations. No monitoring and no restart outside that list. No uncounted screening.

Per start the fixed part is 1 init + 8 calibration + 2 candidate evaluations. The remaining queries are SPSA iterations (two queries each). Because that remainder is not divisible by 8, starts 2 and 3 receive one extra iteration:

| cap | iteration queries by start |
|---:|---|
| 128 | 20, 20, 22, 22 |
| 1024 | 244, 244, 246, 246 |

SPSA is Spall: `A=25`, exponents 0.602 and 0.101, `c=1`, target initial step `0.5σ`, clip `2σ`. σ is γ 0.10, θ 0.25, α 0.02 times the frozen scale, s 0.15, QAOA angles 0.10. The α scale starts at 1 and is halved, at most twice, if a unit step changes `p_leak` by more than 0.01 at the echo init on seed 10008. Selection uses energy only. The ground state is not an init and not a stopping rule.

## Read rule

An energy below `E_+ - 10^{-3} N`, with `E_+ = -B* N`, is an escape from the product state. Absence of such an escape under this budget is evidence about the search, not a proof that real α cannot represent a better state. Paired F and E versus QAOA are reported at both caps, with the initial point, η, leakage, and every failed start.

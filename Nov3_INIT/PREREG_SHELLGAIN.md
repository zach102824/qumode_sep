# N3INIT shell-emphasized SPSA at L=3

Bounded development. Original shared-Uc Nov3 only. Not LE’s two-stage freeze-Uc stream. SPSA only. Energy selection. Fidelity post hoc. No GR patch. No candidacy file. Holdout FAIL on `10144..10151` still stands. Pair-shell REJECT stands. Do not implement paper \(\mathcal{V}(M)\). Do not unshare shells.

Hypothesis: joint SPSA under-updates shells relative to shared \(U_c\) at depth; a larger shell/Uc `a_k/c_k` ratio may recover a Nov3 energy edge at L=3.

## Arms

N=15, L=3 only, `Q=3072` each. Same v4 center-4 equal-block skeleton. Same Hamiltonian, packing, shared \(U_c\) (\(\gamma,\theta_K,\theta_M\)) and shared 3-shell per layer. Circuit is identical for A–C.

| arm | circuit | method_id | shell `a_k` multiplier |
|---|---|---:|---:|
| A `JOINT` | shared-shell Nov3 | 1 | 1 |
| B `SHELL2` | shared-shell Nov3 | 3 | 2 |
| C `SHELL4` | shared-shell Nov3 | 4 | 4 |
| D `qaoa_center4` | QAOA `p=L=3` | 0 | n/a (all coords 1) |

Method ids 3 and 4 are independent SPSA streams. They are not Nov3pair and not Nov9.

## Exact gain vectors (L=3, 12 Nov3 params)

Frozen σ, unchanged: \(\sigma=(\gamma{=}0.10,\,\theta_K{=}0.25,\,\theta_M{=}0.25,\,s{\times}9{=}0.15)\).

Frozen scalar schedule, unchanged: `c=0.1`, `A=25`, exponents `0.602/0.101`, 4 two-sided calibration pairs, `a0=0.5/median(|g_j|/σ_j)` over all coordinates of those pairs (or 0 if that median `<1e-15`), `a_k=a_gain/(k+1+A)^α`, `c_k=c/(k+1)^γ`. Calibration perturbations stay `c·σ·Δ`. Clip stays `2σ`.

Per-coordinate multiplier `a_mult` applies only to the iteration update `clip(a_k · a_mult · g, 2σ)`. `c_k` is not scaled. Therefore shell `a_k/c_k` is exactly `ratio` times the Uc `a_k/c_k`.

- JOINT `a_mult = (1,1,1, 1×9)`
- SHELL2 `a_mult = (1,1,1, 2×9)`
- SHELL4 `a_mult = (1,1,1, 4×9)`
- QAOA `a_mult = (1×6)`

Starts: Nov3 start 0 is `γ=0.25`, `θ=0`, every layer `(s1,s2,s3)=(−0.5,0,0)`. QAOA start 0 is `(γ,β)=(0.25,0.25)` every layer. Starts 1–3 stay in the historical boxes. Four equal blocks, `T=Q/8−6=378`. Inclusive `Q=3072`. Energy selection among init, last iterate, Polyak of last `min(64,T)`, midpoint. Lowest energy; energy tie keeps init; across starts lower `start_id` wins. Fidelity is post hoc.

## Seeds

Exclusive unread band `10368..10375`, n=8. Provenance, verified before any instance or SPSA:

- Not LE two-stage `10360..10367`.
- Not local R25 `10344..10359`.
- Not FT / prior Nov3_INIT `≤10335` (covers train, holdout `10144..10151`, v9, v10, pair-shell `10328..10335`).
- Not validation `11000..11031`.
- Not GR final `20000..20999`.
- No existing `Nov3_INIT/results/**` cell or instance file for these seeds.

The runner refuses every other seed. `B*` is new, classical, interior entropy max on sites `0..floor(N/2)−1`, not a quantum query.

## Grid

New directory `results/shellgain/`. `N=15`, `L=3`, `q=1024`, `Q=3072`. Four arms. Energy-only selection.

If a cell is infeasible, stop that cell with a measured blocker. Do not approximate physics. Fail any cell → reject; no GR.

## Gates (prespecified, energy only, before any SPSA)

Bootstrap 4000, `default_rng(3334)`, seeds resampled. Means are n=8 seed means.

**Winner** = the arm of `{SHELL2, SHELL4}` with the lower mean E (mean-E tie → SHELL2).

**Primary (dev):** winner vs QAOA mean `ΔE = E(winner)−E(QAOA)` CI entirely `< 0`.

**Secondary:** that winner’s mean E `≤` JOINT mean E.

Fail primary or secondary, or any infeasible cell → **REJECT_SHELL_GAIN**. Pass both → **SHELL_GAIN_L3_EDGE**. Either label: no GR patch, no candidacy file.

## Queries

Each cell ledger equals `Q=3072`. Report new shell-gain quantum queries and cumulative total `2301952 + new`. Historical v1–v10 and pair-shell queries stay spent once. Report wall and CPU. Fit workers 4, BLAS 1.

## Off limits

GR PR #77. Validation. Final. Nov9, SYK, AF, N21, paper \(\mathcal{V}(M)\), per-cavity unshare, LE two-stage freeze-Uc, extra agents, candidacy notes, frozen-benchmark retune.

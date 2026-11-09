# N3INIT v3 preregistration

Written after v1 and v2, including both test halves, were read. Seeds `10022..10025` are a third look. Freeze still uses only `10018..10021`.

v2 adopted `center4` at every `q` on the train rule, and the test half agrees. This batch keeps that center and changes only the split of `Q`.

## Arm spread8

Same circuit, `B*`, seeds, `N ∈ {9,12}`, `L ∈ {1,3}`, and frozen SPSA block. Both Nov3 and QAOA are run. Method ids: Nov3 `82`, QAOA `83`.

- Eight equal blocks. `T = Q/16 - 6`, which is `2, 26, 58` at `L=1` and `18, 90, 186` at `L=3`. Ledger `8 * (12 + 2T) = Q`.
- Start 0 is the v2 center: Nov3 `γ=0.25`, `θ=0`, every layer `s1=-0.5`; QAOA every angle `0.25`.
- Starts 1–7 use the v1 equal4 box (`U(-0.5,0.5)` on cost or QAOA angles, `U(-0.3,0.3)` on Nov3 shells). No correlated residual and no single-energy screen.

## Freeze

For each `q`, over the 16 train points, adopt `spread8` over `center4` only if

- `mean(E_Nov3,spread8 - E_Nov3,center4) < 0`, and
- `mean(E_Nov3,spread8 - E_QAOA,spread8) ≤ mean(E_Nov3,center4 - E_QAOA,equal4)`.

`E_QAOA,equal4` is the v1 ledger, already charged, not rerun. New queries are the spread8 blocks for both methods. The test half runs after `FREEZE_V3.json` and does not change it.

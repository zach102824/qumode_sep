# N3INIT v2 preregistration

Written after v1 train and v1 test were both computed and read. The freeze below still uses only seeds `10018..10021`. Seeds `10022..10025` are a second look, not a fresh holdout. v1's `FREEZE.json` is not edited.

## What v1 left confounded

`halve8to2` changed three things at once: the Nov3 deterministic center (`s1=-0.5` on every layer), correlated random draws, and an 8-screen / 2-finalist split. The train rule adopted it at `q=128` and `q=512` and kept `equal4` at `q=1024`. This batch removes two of those changes.

## Arm center4

Same circuit, same `B*`, same seeds, same `N ∈ {9,12}`, `L ∈ {1,3}`, `Q = q L`, same frozen SPSA block as v1 `equal4` (`T = Q/8 - 6`).

- Nov3 method id stays `1`, so starts 1–3 and their SPSA noise match v1 `equal4` exactly. A cell is a failure if those three start energies differ from the stored v1 tails by more than `1e-8`.
- Nov3 start 0 is the only new draw: `γ=0.25`, `θ_K=θ_M=0`, every layer `(s1,s2,s3)=(-0.5, 0, 0)`.
- QAOA is not rerun. The paired baseline is v1 `equal4` QAOA, already charged in the v1 ledger. Those queries are not added again and are not treated as free. New charged queries are the Nov3 `center4` blocks only.

No halving, no correlated residual, no gain retune, no L1→L3 transfer.

## Freeze

For each `q`, over the 16 train points, adopt `center4` only if

- `mean(E_center4 - E_equal4) < 0` for Nov3, and
- `mean(E_center4 - E_qaoa) ≤ mean(E_equal4 - E_qaoa)`,

where `E_qaoa` is v1 `equal4` QAOA. Otherwise the recommendation stays `equal4`. The test half is run after `FREEZE_V2.json` and does not change it.

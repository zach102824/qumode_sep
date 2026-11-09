# N3INIT L=3 spread8 restart allocation

Bounded development. Original shared-Uc Nov3 only. SPSA only. Energy selection. Fidelity post hoc. No GR patch. No candidacy file. Holdout FAIL, pair-shell REJECT, and shell-gain REJECT still stand.

Not more shared-shell center-4 L2 sampling. Not LE N18 depth `10320..10327`. Not local R25 N18 L1 `10344..10359`. Not LE two-stage freeze-Uc. Not paper \(\mathcal{V}(M)\). Not per-cavity unshare.

Hypothesis: at L=3, four long equal4 blocks miss better random starts; eight shorter SPSA blocks at the same inclusive `Q` may recover a Nov3 energy edge.

## Arms

N=15, L=3 only, `Q=3072` each. Same frozen σ, gains, center-4 start 0, energy selection. Circuit is original shared-shell Nov3 (`3+3L=12` params) or QAOA `p=L=3` (`2L=6`).

| arm | circuit | allocation | method_id | T |
|---|---|---|---:|---:|
| A `nov3_equal4` | shared-shell Nov3 | 4 equal blocks | 1 | 378 |
| B `nov3_spread8` | shared-shell Nov3 | 8 equal blocks | 5 | 186 |
| C `qaoa_equal4` | QAOA p=3 | 4 equal blocks | 0 | 378 |
| D `qaoa_spread8` | QAOA p=3 | 8 equal blocks | 6 | 186 |

`T = Q/8−6` (equal4) or `T = Q/16−6` (spread8). Inclusive ledger equals `Q` in both recipes. Start 0 is the v4 coherent center. Starts 1–3 (equal4) or 1–7 (spread8) stay in the historical boxes. Distinct method ids so streams are independent.

## Seeds

Exclusive unread band `10376..10383`, n=8, taken from the keep-alive window `10376..10391`. Provenance, verified before any instance or SPSA:

- Not LE two-stage / LE-reserved `10360..10375`.
- Not local R25 `10344..10359`.
- Not local-reserved `≥10392`.
- Not FT / prior Nov3_INIT `≤10335` (includes pair-shell `10328..10335`).
- Not holdout `10144..10151`.
- Not validation `11000..11031`.
- Not GR final `20000..20999`.
- Window remainder `10384..10391` left unread.
- No existing `Nov3_INIT/results/**` file for `10376..10383`.

The runner refuses every other seed. `B*` is new, classical, interior entropy max on `0..floor(N/2)−1`, not a quantum query.

## Grid

New directory `results/spread8/`. Energy-only selection. If a cell is infeasible, stop that cell with a measured blocker. Fail any cell → reject; no GR.

## Gates (prespecified, energy only, before any SPSA)

Bootstrap 4000, `default_rng(3334)`, seeds resampled.

**Primary (dev, fair allocation):** `nov3_spread8` vs `qaoa_spread8` mean `ΔE` CI entirely `< 0`.

**Secondary:** `nov3_spread8` mean E `≤` `nov3_equal4` mean E (eight short blocks not worse than four long ones).

Fail primary or secondary, or any infeasible cell → **REJECT_SPREAD8_L3**. Pass both → **SPREAD8_L3_EDGE**. Either label: no GR patch, no candidacy file.

equal4 Nov3 vs equal4 QAOA is reported as context, not a gate.

## Queries

Each cell ledger equals `Q=3072`. Report new queries and cumulative total `2400256 + new`. Historical v1–v10, pair-shell, and shell-gain stay spent once. Fit workers 4, BLAS 1.

## Off limits

GR PR #77. Validation. Final. Nov9, SYK, AF, N21, paper \(\mathcal{V}(M)\), per-cavity shells, extra L2 center-4 sampling, extra agents, candidacy notes, frozen-benchmark retune.

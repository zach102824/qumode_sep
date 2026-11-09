# Unread-seed N=15 and N=18 L=1 width panel

Bounded development. Original shared-Uc Nov3 vs QAOA `p=L`. Frozen v4 center-4 SPSA. Energy selects. F post hoc. Entropy-peak \(B^*\) only. **Not** L=2 sampling, **not** an L=3 schedule/architecture tweak, **not** compiler, **not** ε, **not** GR, **not** Nov9/SYK, **not** a gauge redraw.

v5 already saw an N=18 L=1 Nov3−QAOA energy edge, but on the **same** development seeds `10018..10025` used at N=15. Width and instance identity were confounded. PR112 showed the N=15 L=1 edge is not a spin-flip labeling artifact on `10552..10559`. This panel asks whether exclusive unread instances still show the **shallow** edge at both widths.

Holdout FAIL on L=2 (`10144..10151`) still stands. Hardware \(W_L\) remains unvalidated. Compiler stream closed (PR111 `WCOMP_WS_FAIL_BUDGET`).

## Seeds (frozen assignment, before fits)

Exclusive unread `10568..10583` (n=16). Collision-checked vs LE `10432..39` and `10552..67`, local R31 `10536..51`, R30 `10504..19`, FT `10520..35`, V(M) `10480..87`, anneal `10456..63`, spread8 `10376..83`, shell-gain `10368..75`, holdout `10144..51`, validation `11000..11031`, GR/final `20000..20999`.

| N | seeds | n | L | Q |
|---:|---|---:|---:|---:|
| 15 | `10568..10575` | 8 | 1 | 1024 |
| 18 | `10576..10583` | 8 | 1 | 1024 |

Assignment is by seed order, not by outcome. Do not pool the 16 as iid across N. Do not move a seed to the other N after seeing energies. One budget (`q=1024`); not a q-sweep.

## Recipe

Exact v4 `center4`: Nov3 start 0 \(\gamma=0.25\), \(\theta_K=\theta_M=0\), \((s_1,s_2,s_3)=(-0.5,0,0)\); QAOA start 0 \((\gamma,\beta)=(0.25,0.25)\); starts 1–3 historical boxes; Spall \(c=0.1\) \(A=25\) \(0.602/0.101\); four equal blocks; method ids 1 and 0. `make_rng(seed, N, B*, L, method_id, start_id, stream)`.

\(B^*\) is the interior entropy max on `0 .. floor(N/2)−1` (N=15: `0..6`; N=18: `0..8`). Classical, not a quantum query. Interleaved packing \(m=N/3\).

## Resource gate (conformance, before instances)

Smoke on published seed `10018` at N=18 only: one sparse `eigsh` at \(b=0.5\), one Nov3 L=1 energy at the frozen center. Gate if eigsh `< 30` s, energy `< 0.5` s, peak RSS `< 4` GiB, `dim=2^{18}`, no dense H. That seed is **not** fitted here. If the gate fails: stop, no SPSA.

## Analysis (exploratory, no adoption)

For each N separately, paired \(\Delta E_s=E_{\mathrm{Nov3}}-E_{\mathrm{QAOA}}\) on that N’s 8 instances. Bootstrap 4000, `default_rng(3334)`, seeds resampled. Report mean, CI, wins, post-hoc F, \((E-E_0)/N\), start-0 counts.

- Both N: mean \(\Delta E<0\) and CI entirely `<0\) → **WIDTH_L1_BOTH**
- Only N=15 → **WIDTH_L1_N15_ONLY**
- Only N=18 → **WIDTH_L1_N18_ONLY**
- Neither, or any infeasible cell → **WIDTH_L1_MISS** (or blocked)

These are descriptive flags on this band, **not** an adoption gate, **not** a GR patch, **not** a population claim. Do not interpret N=18 success as undoing L=2 holdout FAIL.

## Queries

32 cells × 1024 = **32768** energy queries. Historical SK **2883584**. Cumulative **2916352** if complete. Classical \(B^*\) separate. Workers 4, BLAS 1.

## Off limits

L=2, L=3, extra q, extra K/α, pair-shell, V(M), compiler, ε, GR, validation, final, Nov9/SYK, extra agents.

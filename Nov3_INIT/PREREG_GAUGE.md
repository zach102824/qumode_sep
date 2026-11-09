# SK spin-flip gauge robustness of Nov3 L=1 (exploratory)

Bounded development. Original shared-Uc Nov3 vs QAOA `p=L`. Frozen v4 center-4 SPSA. Energy selects. F post hoc. Same entropy-peak SK instances (`B*` interior max, cut `0..floor(N/2)−1`). **No new model, no ansatz retune, no GR, no compiler, no ε/control/depth sweep.**

Scientific question: is the shallow N=15 L=1 Nov3−QAOA energy edge stable under **exact gauge-equivalent labelings** of each SK Hamiltonian, or tied to the presented Fock/bit labeling?

Compiler stream is closed (PR109 history, PR111 `WCOMP_WS_FAIL_BUDGET`). Hardware implementation of \(W_L\) remains unvalidated.

## Gauge (frozen before any SPSA)

Local spin-flip \(g_i=\pm 1\). Identity plus three deterministic random gauges, **never selected on energy**.

\[
J'_{ij}=g_i g_j J_{ij},\qquad
H'=\sum_{i<j}J'_{ij}Z_i Z_j-B^*\sum_i X_i.
\]

The uniform transverse field is **not** gauged. Exact conjugation: \(U=\prod_{g_i=-1}X_i\) with bit \(i\) as \((\mathrm{idx}\gg i)\&1\). Then \(H'=U H U^\dagger\), \(|\mathrm{gs}'\rangle=U|\mathrm{gs}\rangle\). Entropy \(S_A\) is invariant under local unitaries on a fixed partition, so **reuse original \(B^*\)**; do not scan or select a new field. Map the stored ground state by the same \(U\).

RNG (separate from SPSA): `SeedSequence([4401, seed, N, gid])`. `gid=0` is all \(+1\). `gid=1,2,3` are independent Rademacher \(\pm 1\). Vectors are written to `results/gauge/GAUGE.json` in conformance, hashed, and not edited after seeing energies.

SPSA/init RNG is **unchanged across gauges**: `make_rng(seed, N, B*, L, method_id, start_id, stream)` with method ids 1 (Nov3) and 0 (QAOA). Gauge id is not in that tuple. Same \(x_0\) and same SPSA noise for a given `(seed, method)` on every gauge. No hand-picked gauge or parameter repair favoring Nov3.

## Seeds

Exclusive `10552..10559`, n=8. Verified before instance or SPSA:

- Not local R31 `10536..10551`
- Not FT WL `10528..10535` or cutoff `10520..10527` or R30 `10504..10519`
- Not FT \(\le 10535\), LE `10432..10439`, holdout `10144..10151`
- Not validation `11000..11031`, not GR/final `20000..20999`
- No existing `Nov3_INIT/results/**` file for these seeds

## Grid

`results/gauge/`. N=15, L=1, `Q=1024` **per method per representation**. Frozen v4 `center4` / derived QAOA `(γ,β)=(0.25,0.25)`. Logical Hilbert space (no Fock, no compiled \(W\)).

| representation | methods | method_id | Q |
|---|---|---:|---:|
| gid 0 identity, gid 1–3 frozen random | Nov3rd, QAOA | 1, 0 | 1024 each |

64 arms. Inclusive energy queries \(8\times 4\times 2\times 1024=65536\). Classical \(B^*\) and gauge-algebra checks are **not** energy queries.

## Conformance (must pass before SPSA)

On a toy instance: spectrum / ground energy of \(H'\) equals \(H\); mapped ground is an eigenvector; plus-state energy and \(U|+\rangle=|+\rangle\); QAOA energy **and** fidelity invariant at the derived center and at a frozen random angle. If any fail: report and fix, do not fit.

QAOA exact gauge covariance is an **implementation sanity check**, not silently assumed. After fits, QAOA selected energies across the four gauges of a seed should agree to numerical tolerance (\(\max|\Delta E|\le 10^{-8}\)). Failure → `GAUGE_IMPL_FAIL` (bug), still report the numbers, still stop.

Nov3 is **not** assumed covariant (SNAP shells and packing live in the bit labels). If Nov3 energies vary across gauges while QAOA does not, that is circuit expressivity / labeling, not optimizer RNG (the SPSA seed is shared).

## Analysis (frozen before fits; exploratory, no adoption)

Gauge variants of one SK instance are **correlated**. The statistical unit is the **8 original instances**, not 32 independent samples.

For instance \(s\) and gauge \(g\), \(\Delta E_{s,g}=E_{\mathrm{Nov3}}(s,g)-E_{\mathrm{QAOA}}(s,g)\).

**Primary:** \(\bar\Delta_s=\frac14\sum_{g=0}^{3}\Delta E_{s,g}\). Mean of \(\bar\Delta_s\) over the 8 instances, paired bootstrap 4000, `default_rng(3334)`, instances resampled. Report CI. **Not** an adoption gate, **not** a multiple-test population claim, **not** a GR patch.

Also report, all of them, no cherry-pick:

- Identity-only (\(g=0\)) mean \(\Delta E\) and CI (same 8 instances)
- Each fixed \(g\in\{1,2,3\}\) mean \(\Delta E\) and CI
- Per-instance range \(\max_g\Delta E_s-\min_g\Delta E_s\) and the worst observed gauge for Nov3 (largest \(\Delta E_{s,g}\), i.e. weakest Nov3-vs-QAOA edge)
- QAOA cross-gauge \(\max|E_g-E_0|\) (sanity)
- Nov3 cross-gauge energy range (expressivity)
- F post hoc on every arm

Stop after this panel. No extra gauge, no L>1, no new seeds.

## Queries

Each cell ledger equals `Q=1024`. New energy queries 65536. Historical SK **2818048**. Cumulative **2883584** if every cell completes. Fit workers 4, BLAS 1.

## Off limits

GR PR #77. Validation. Final. Nov9, SYK, AF, N21, compiler retune, extra \(K\)/\(\alpha\), paper \(\mathcal{V}(M)\), extra agents, candidacy notes, frozen-benchmark retune.

# Joint-parity gate (`jp`) fleets at 200 SPSA steps

Updated Asia/Shanghai: 2026-09-28 13:25 CST (λ=0/3 artifacts created 04:17–04:26 UTC;
λ=1/2 follow-up artifacts created 05:10–05:19 UTC).

## Gate definition

`jp` is a new fixed bus unitary on A⊗B (identity on transmons d, e), added in
`noiseless/unitaries.py` as `joint_parity_ab()` and registered as `"jp"` in
`U_NAMES` / `build_fixed_u`:

\[
\mathrm{jp}\,|n,m\rangle = (-i)^{(n+m)\bmod 2}\,|n,m\rangle
= e^{-i\pi/4}\,\exp\!\big(+i\tfrac{\pi}{4}\,P_A P_B\big)\,|n,m\rangle,
\qquad P_{A,B} = (-1)^{\hat n_{A,B}} .
\]

It is a ZZ(π/2) rotation on the two cavity *parity qubits*: even joint
parity picks up no phase, odd joint parity picks up −i.

**Relation to `cz_nm`.** With the local (single-cavity) parity phase
`L = diag(i^{n mod 2})`,

\[
(L_A \otimes L_B)\,\mathrm{jp} = \mathrm{cz\_nm},\qquad
\text{i.e. } \mathrm{jp} = (L_A\otimes L_B)^\dagger\,\mathrm{cz\_nm},
\]

because `i^{p+q} (-i)^{(p+q) mod 2} = (-1)^{pq}` for parities p, q ∈ {0,1}.
Verified numerically on the full 64×64 A⊗B block (NFOCK=8) and on the full
256-dim space: max |(L_A⊗L_B)·jp − cz_nm| = 0.0 (tolerance 1e-12); `jp` is
exactly diagonal and unitary. A unit test (`test_joint_parity_actions`) was
added. So `jp` and `cz_nm` are locally equivalent, but the local correction
`L` is a parity-dependent phase that the local ECD layers do not generate for
free, so the two gates give different ansatz landscapes.

**Physical motivation.** One coupler transmon dispersively coupled to both
cavities with equal shifts χ_A = χ_B = χ. (1) Map joint parity onto the
coupler: π/2 on the coupler, wait t = π/χ (the coupler acquires phase
π(n+m)), π/2 back, so the coupler ends in |g⟩ for even and |e⟩ for odd
(n+m). (2) Apply S† on the coupler (phase −i on |e⟩). (3) Map again (the
inverse sequence) to return the coupler to |g⟩ and disentangle it. Net
effect: `(-i)^{(n+m) mod 2}` on the cavities. Unlike `cz_nm`
((−1)^{nm}), this needs no χ_AB cross-Kerr or conditional-on-both-parities
logic, only a single joint-parity map with matched χ's.

Note: `jp` is appended *last* in `U_NAMES` (index 11) so the index-based
per-trial seeds of all existing gates are unchanged. Consequently `jp`
trials use different random initialisations than the `cz_nm` trials, even
with the same `--seed`.

## Settings

Identical to the `cz_nm` fleets in `CZ_NM_SOFTCAP_SUMMARY.md`: 4 layers
(L*=4), 20 unique-ground-state `four_sat` Hamiltonians × 25 trials = 500
trials, 200 SPSA steps, seed `20260917`, `--lambda1 0`, no `--adapt-lambda`,
workers=1, `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`. Run
sequentially on the 1-CPU box (~4.5 min each). Both arms: 500/500 OK, zero
failures.

```bash
python noiseless/run_u_sweep.py --u-names jp --layers 4 --trials 25 --steps 200 \
  --workers 1 --seed 20260917 --lambda1 0 --tag jp_B0
python noiseless/run_u_sweep.py --u-names jp --layers 4 --trials 25 --steps 200 \
  --workers 1 --seed 20260917 --lambda1 0 --lambda 3 --beta-max 2.1 --tag jp_lam3
```

The λ=3 arm uses the fixed penalty `3 · sum_i max(|β_i| − 2.1, 0)^2`.
“Success” = the most likely final bitstring equals the unique ground state;
p(GS) = final probability of the ground-state bitstring (same
`run_u_sweep.py` aggregation as all prior summaries). “Mean trial-max” is
the mean over trials of each trial's maximum |β|.

## Results

| arm | jobs | success | mean p(GS) | mean\|β\| | median\|β\| | mean trial-max\|β\| | per-H success min / median / max | #H at 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **jp Gibbs** | 500/500 | **0.932** (466/500) | **0.1717** | 1.8475 | 1.8250 | 3.3882 | 0.84 / 0.92 / 1.00 | 5/20 |
| **jp λ=3** | 500/500 | **0.736** (368/500) | **0.1163** | 1.5593 | 1.5574 | 2.1277 | 0.36 / 0.76 / 1.00 | 2/20 |
| cz_nm Gibbs | 500/500 | 0.922 (461/500) | 0.1663 | 1.8517 | 1.8444 | 3.3913 | 0.72 / 0.96 / 1.00 | 8/20 |
| cz_nm λ=3 | 500/500 | 0.790 (395/500) | 0.1221 | 1.5467 | 1.5529 | 2.1245 | 0.56 / 0.80 / 1.00 | 2/20 |
| ck_pi4 Gibbs | 500/500 | 0.940 (470/500) | 0.1620 | 1.8441 | 1.8378 | 3.3780 | 0.80 / 0.96 / 1.00 | 5/20 |
| ck_pi4 λ=3 | 500/500 | 0.810 (405/500) | 0.1315 | 1.5509 | 1.5593 | 2.1229 | 0.56 / 0.80 / 1.00 | 2/20 |

cz_nm and ck_pi4 rows are recomputed from the existing summary JSONs
(`fleet_cznm_B0_gibbs_steps200_*`, `fleet_cznm_F3_lam3_bmax2p1_steps200_*`,
`fleet_adapt_B0_gibbs_20260924T032122Z_*`,
`fleet_fixed_lam3_bmax2p1_steps200_*`) and match the earlier summaries.

### Per-instance successes (out of 25), four_sat_000 … four_sat_019

| arm | per-H n_success |
|---|---|
| jp Gibbs | 23 24 25 22 22 25 23 25 25 23 21 24 23 23 22 21 23 23 24 25 |
| cz_nm Gibbs | 24 25 24 21 22 23 25 25 25 25 18 25 22 24 25 20 19 21 23 25 |
| jp λ=3 | 9 22 21 12 15 22 18 25 25 20 13 21 18 20 19 19 12 15 18 24 |
| cz_nm λ=3 | 16 22 20 16 16 24 21 25 25 20 15 18 19 16 22 22 19 14 21 24 |

- Gibbs: jp better / equal / worse than cz_nm on 9 / 4 / 7 instances. The
  hardest instance for both is `four_sat_010` (jp 21/25 vs cz_nm 18/25).
- λ=3: jp better / equal / worse on 4 / 5 / 11 instances; the worst jp
  instance is `four_sat_000` (9/25, vs 16/25 for cz_nm).

## λ scan for jp (λ = 0, 1, 2, 3; β_max = 2.1)

Follow-up: two more `jp` fleets with settings identical to `jp_lam3` except λ
(500/500 OK, zero failures each):

```bash
python noiseless/run_u_sweep.py --u-names jp --layers 4 --trials 25 --steps 200 \
  --workers 1 --seed 20260917 --lambda1 0 --lambda 1 --beta-max 2.1 --tag jp_lam1
python noiseless/run_u_sweep.py --u-names jp --layers 4 --trials 25 --steps 200 \
  --workers 1 --seed 20260917 --lambda1 0 --lambda 2 --beta-max 2.1 --tag jp_lam2
```

`cz_nm` and `ck_pi4` rows are taken from existing summary JSONs at the λ
values they have (0, 2, 3); the args (gate, L*=4, 25 trials, 200 steps, seed,
λ, β_max) were checked programmatically for every row. λ=0 means Gibbs-only
(no cap term).

| gate | λ | success | mean p(GS) | mean\|β\| | mean trial-max\|β\| | per-H success min / median / max | #H at 100% | source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **jp** | 0 | 0.932 (466/500) | 0.1717 | 1.8475 | 3.3882 | 0.84 / 0.92 / 1.00 | 5/20 | `jp_B0_20260928T042134Z_summary.json` |
| **jp** | 1 | 0.788 (394/500) | 0.1222 | 1.5729 | 2.2849 | 0.52 / 0.80 / 1.00 | 3/20 | `jp_lam1_20260928T051459Z_summary.json` |
| **jp** | 2 | 0.774 (387/500) | 0.1174 | 1.5696 | 2.1705 | 0.52 / 0.76 / 1.00 | 4/20 | `jp_lam2_20260928T051924Z_summary.json` |
| **jp** | 3 | 0.736 (368/500) | 0.1163 | 1.5593 | 2.1277 | 0.36 / 0.76 / 1.00 | 2/20 | `jp_lam3_20260928T042559Z_summary.json` |
| cz_nm | 0 | 0.922 (461/500) | 0.1663 | 1.8517 | 3.3913 | 0.72 / 0.96 / 1.00 | 8/20 | `fleet_cznm_B0_gibbs_steps200_20260924T062202Z_summary.json` |
| cz_nm | 2 | 0.758 (379/500) | 0.1208 | 1.5575 | 2.1616 | 0.56 / 0.74 / 1.00 | 1/20 | `fleet_cznm_F2_lam2_bmax2p1_steps200_20260924T062631Z_summary.json` |
| cz_nm | 3 | 0.790 (395/500) | 0.1221 | 1.5467 | 2.1245 | 0.56 / 0.80 / 1.00 | 2/20 | `fleet_cznm_F3_lam3_bmax2p1_steps200_20260924T063050Z_summary.json` |
| ck_pi4 | 0 | 0.940 (470/500) | 0.1620 | 1.8441 | 3.3780 | 0.80 / 0.96 / 1.00 | 5/20 | `fleet_adapt_B0_gibbs_20260924T032122Z_summary.json` |
| ck_pi4 | 2 | 0.834 (417/500) | 0.1352 | 1.5583 | 2.1653 | 0.60 / 0.84 / 1.00 | 2/20 | `fleet_fixed_lam2_bmax2p1_steps200_20260924T033612Z_summary.json` |
| ck_pi4 | 3 | 0.810 (405/500) | 0.1315 | 1.5509 | 2.1229 | 0.56 / 0.80 / 1.00 | 2/20 | `fleet_fixed_lam3_bmax2p1_steps200_20260924T033612Z_summary.json` |

Amplitude cut for jp relative to jp λ=0 (mean\|β\| / mean trial-max\|β\|):
λ=1 −14.9% / −32.6%, λ=2 −15.0% / −35.9%, λ=3 −15.6% / −37.2%. Tail of the
per-trial max\|β\| from the full dumps (box only): fraction of trials with
max\|β\| > 2.1 is 0.96 / 0.90 / 0.85 / 0.79 and the worst trial-max is
6.10 / 2.85 / 2.76 / 2.39 for λ = 0 / 1 / 2 / 3.

### Per-instance successes for jp (out of 25), four_sat_000 … four_sat_019

| λ | per-H n_success |
|---:|---|
| 0 | 23 24 25 22 22 25 23 25 25 23 21 24 23 23 22 21 23 23 24 25 |
| 1 | 13 21 20 16 15 23 17 25 25 22 19 22 19 20 20 22 14 18 18 25 |
| 2 | 14 21 22 15 13 25 17 25 25 21 18 20 19 16 19 23 15 16 18 25 |
| 3 | 9 22 21 12 15 22 18 25 25 20 13 21 18 20 19 19 12 15 18 24 |
| cz_nm λ=2 | 15 24 21 18 14 22 19 24 25 20 14 18 18 20 18 21 14 15 15 24 |

- The same instances are hard at every λ > 0: `four_sat_000`, `003`, `004`,
  `016` (and `006`); `007`, `008`, `019` stay at or near 25/25.
- λ=3's extra losses are concentrated on `000` (9 vs 13–14 at λ=1/2), `003`,
  `010` and `016`; its per-instance floor (0.36) is well below λ=1/2 (0.52).
- jp λ=2 vs cz_nm λ=2: jp better / equal / worse on 13 / 1 / 6 instances.

### λ-scan interpretation

- **Does λ=1 keep more success while still cutting |β|?** Yes, modestly.
  λ=1 has the highest capped-arm success (0.788) and p(GS) (0.122), and gets
  essentially the same mean\|β\| cut as λ=2/3 (−14.9% vs −15.0/−15.6%). What
  it gives up is the tail: mean trial-max 2.28 (vs 2.17 / 2.13) and more
  trials overshooting 2.1 (90% vs 85% / 79%). The λ=1 vs λ=2 success gap
  (+1.4 pp) is within noise (≈0.5σ); λ=1 vs λ=3 (+5.2 pp) is ≈1.9σ.
- **Best trade-off for jp: λ=1–2, with λ=2 the balanced choice.** λ=2 keeps
  success statistically tied with λ=1 (0.774) while holding trial-max close
  to the cap (2.17). Choose λ=1 if mean\|β\| is what matters, and λ=3 only if
  the hard tail matters: it is the only arm with trial-max ≈ 2.13, and it
  costs another ~4–5 pp of success plus a weaker worst instance.
- **Versus the other gates:** jp's success falls with λ (0.788 → 0.774 →
  0.736), unlike cz_nm (0.758 at λ=2 < 0.790 at λ=3). At λ=2 jp slightly
  beats cz_nm (+1.6 pp, ≈0.6σ, 13/20 instances better). jp λ=1 matches
  cz_nm's best capped arm (λ=3, 0.790). ck_pi4 is still the best
  capped gate: jp λ=2 is −6.0 pp below ck_pi4 λ=2 (≈2.4σ), and jp λ=1 is
  −4.6 pp below it (≈1.9σ). With the cap on, all three gates reach the same
  mean\|β\| ≈ 1.55–1.57 at a given λ.

## Interpretation (λ = 0 vs 3, original runs)

- **Gibbs-only:** `jp` is statistically indistinguishable from `cz_nm`
  (0.932 vs 0.922, Δ = +1.0 pp, ≈0.6σ) and from `ck_pi4` (0.940), with the
  highest mean p(GS) of the three (0.172). It has a *tighter* per-instance
  floor (worst instance 0.84 vs 0.72 for cz_nm) but fewer perfect instances
  (5 vs 8), i.e. it is more uniform across Hamiltonians. Amplitudes are
  unchanged (mean |β| ≈ 1.85, trial-max ≈ 3.39).
- **Soft cap λ=3, β_max=2.1:** the cap achieves the same amplitude cut as for
  the other gates (mean |β| 1.56, trial-max 2.13, ~16% / ~37% below jp
  Gibbs), but `jp` loses more success: 0.736 vs 0.790 (cz_nm) and 0.810
  (ck_pi4). The jp–cz_nm gap is −5.4 pp (≈2σ, marginal), driven by a few
  instances (000, 016, 003) with a worst-case of 9/25.
- **Bottom line:** the joint-parity gate, which is plausibly easier to
  realise than `cz_nm` (single coupler with χ_A = χ_B, no cross-Kerr),
  matches `cz_nm`/`ck_pi4` in the unconstrained Gibbs setting, so the
  missing local parity phases L_A⊗L_B cost nothing there. Under the |β| soft
  cap it is somewhat worse; the local phase correction apparently matters
  more when displacement amplitudes are constrained.

## Artifacts

- `jp_B0_20260928T042134Z_summary.json`
- `jp_lam3_20260928T042559Z_summary.json`
- `jp_lam1_20260928T051459Z_summary.json`
- `jp_lam2_20260928T051924Z_summary.json`
- full dumps `jp_{B0,lam1,lam2,lam3}_*.json` (gitignored, on the box only);
  logs `logs/jp_{B0,lam1,lam2,lam3}.log` (untracked)
- `JP_GATE_SUMMARY.md`

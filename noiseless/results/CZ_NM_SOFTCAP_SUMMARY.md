# Parity-CZ (`cz_nm`) fixed soft-cap fleets at 200 SPSA steps

Updated Asia/Shanghai: 2026-09-24 14:31 CST (run artifacts created 06:22–06:30 UTC).

All three sequential fleets used the same setup: `--u-names cz_nm`, four layers,
20 unique-ground-state `four_sat` Hamiltonians, 25 trials per Hamiltonian,
200 SPSA steps, seed `20260917`, `--lambda1 0`, no `--adapt-lambda`, and
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`. The box had one CPU,
so the practical worker count was 1. Every arm completed 500/500 jobs with zero
failures.

For F2/F3, the cost used the fixed penalty
`lambda * sum_i max(|beta_i|-2.1, 0)^2`. “Mean trial-max” is the mean, over
trials, of each trial's maximum `|beta|`.

## Results

The first cut column compares each arm with this `cz_nm` B0 mean `|beta|` of
1.851652. The final cut column uses the historical reference 1.84.

| arm | lambda | beta max | jobs | success | mean p(GS) | mean\|beta\| | median\|beta\| | mean trial-max\|beta\| | \|beta\| cut vs cz_nm B0 | cut vs 1.84 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 Gibbs | 0 | — | 500/500 | 0.922 (461/500) | 0.1663 | 1.8517 | 1.8444 | 3.3913 | 0.0% | -0.6% |
| F2 | 2 | 2.1 | 500/500 | 0.758 (379/500) | 0.1208 | 1.5575 | 1.5465 | 2.1616 | 15.9% | 15.4% |
| F3 | 3 | 2.1 | 500/500 | 0.790 (395/500) | 0.1221 | 1.5467 | 1.5529 | 2.1245 | 16.5% | 15.9% |

F3 gives the tightest amplitudes: versus this run's B0 it reduces mean
`|beta|` by 16.5% and mean trial-max `|beta|` by 37.4%, while F2 reduces them
by 15.9% and 36.3%, respectively. F2/F3 trade away success and mean p(GS)
relative to B0; F3 recovers 3.2 percentage points of success versus F2.

## Context: prior `ck_pi4` fleets

The prior fixed-soft-cap and adaptive summaries used `ck_pi4` with the same
20-Hamiltonian, 25-trial, 200-step design. Their reported reference numbers
were:

| prior arm | success | mean\|beta\| |
|---|---:|---:|
| B0 Gibbs | 0.940 | ~1.844 |
| lambda=2, beta max 2.1 | 0.834 | 1.558 |
| lambda=3, beta max 2.1 | 0.810 | 1.551 |

Thus the `cz_nm` B0 success (0.922) is below the prior `ck_pi4` B0 (0.940),
while its fixed-cap arms have lower success (F2 0.758, F3 0.790) but very
similar amplitude reductions (mean `|beta|` about 1.56 and 1.55). See
`FIXED_SOFTCAP_SUMMARY.md` and `ADAPT_LAMBDA_SUMMARY.md` for the prior details.

## Artifacts

- `fleet_cznm_B0_gibbs_steps200_20260924T062202Z_summary.json`
- `fleet_cznm_F2_lam2_bmax2p1_steps200_20260924T062631Z_summary.json`
- `fleet_cznm_F3_lam3_bmax2p1_steps200_20260924T063050Z_summary.json`
- `CZ_NM_SOFTCAP_SUMMARY.md`

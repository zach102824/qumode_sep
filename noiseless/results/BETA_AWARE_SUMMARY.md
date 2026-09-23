# β-aware fleet summary (ck_pi4)

Updated UTC: 2026-09-23T09:38:15Z

Baseline (`fleet_phase_bakeoff` ck_pi4 L*=4): success=0.94, mean|β|≈1.84.

## Results

| tag | λ1 | λ3 | β_max | L* | success | mean p(GS) | mean\|β\| | median\|β\| | mean trial-max\|β\| | % over β_max | \|β\| cut vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fleet_beta_aware_A0_ckpi4` | 0 | 0 | — | 4 | 0.940 | 0.1620 | 1.844 | 1.862 | 3.378 | 0.0% | -0.2% |

## Notes

- A0 sanity success=0.940 (baseline 0.94).

## How to toggle

Defaults (`--lambda1 0 --lambda3 0`, no `--beta-max`) preserve Gibbs-only cost.
Enable β terms via CLI on `python -m noiseless.run_u_sweep`.


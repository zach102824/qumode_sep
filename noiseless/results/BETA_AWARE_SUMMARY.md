# β-aware fleet summary (ck_pi4)

Updated UTC: 2026-09-23T09:42:53Z

Baseline (`fleet_phase_bakeoff` ck_pi4 L*=4): success=0.94, mean|β|≈1.84.

## Results

| tag | λ1 | λ3 | β_max | L* | success | mean p(GS) | mean\|β\| | median\|β\| | mean trial-max\|β\| | % over β_max | \|β\| cut vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fleet_beta_aware_A0_ckpi4` | 0 | 0 | — | 4 | 0.940 | 0.1620 | 1.844 | 1.862 | 3.378 | 0.0% | -0.2% |
| `fleet_beta_aware_A1_l1_0p005` | 0.005 | 0 | — | 4 | 0.938 | 0.1622 | 1.831 | 1.842 | 3.342 | 0.0% | 0.5% |
| `fleet_beta_aware_A1_l1_0p01` | 0.01 | 0 | — | 4 | 0.944 | 0.1627 | 1.821 | 1.838 | 3.337 | 0.0% | 1.0% |
| `fleet_beta_aware_A1_l1_0p02` | 0.02 | 0 | — | 4 | 0.930 | 0.1651 | 1.799 | 1.824 | 3.312 | 0.0% | 2.3% |
| `fleet_beta_aware_A1_l1_0p05` | 0.05 | 0 | — | 4 | 0.910 | 0.1638 | 1.757 | 1.777 | 3.258 | 0.0% | 4.5% |
| `fleet_beta_aware_A1_l1_0p1` | 0.1 | 0 | — | 4 | 0.866 | 0.1586 | 1.664 | 1.679 | 3.186 | 0.0% | 9.6% |
| `fleet_beta_aware_A1_l1_0p2` | 0.2 | 0 | — | 4 | 0.774 | 0.1520 | 1.496 | 1.490 | 3.029 | 0.0% | 18.7% |

## Notes

- A0 sanity success=0.940 (baseline 0.94).
- A1 λ1=0.005: success=0.938, mean|β|=1.831, cut=0.5%.
- A1 λ1=0.01: success=0.944, mean|β|=1.821, cut=1.0%.
- A1 λ1=0.02: success=0.930, mean|β|=1.799, cut=2.3%.
- A1 λ1=0.05: success=0.910, mean|β|=1.757, cut=4.5%.
- A1 λ1=0.1: success=0.866, mean|β|=1.664, cut=9.6%.
- A1 λ1=0.2: success=0.774, mean|β|=1.496, cut=18.7%.
- Stopping A1 escalation: success 0.774 < 0.85 at λ1=0.2.
- No λ1 with success≥0.90 AND |β| cut≥15% → skip A2.
- No setting met success≥0.90 with ≥15% |β| cut; see table for tradeoffs.

## How to toggle

Defaults (`--lambda1 0 --lambda3 0`, no `--beta-max`) preserve Gibbs-only cost.
Enable β terms via CLI on `python -m noiseless.run_u_sweep`.


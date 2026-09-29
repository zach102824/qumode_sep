# jp: Gray cavity encoding and layer growth (noiseless, Gibbs-only SPSA)

Updated Asia/Shanghai: 2026-09-29 16:55 CST. Fleets run 16:40–16:49 CST (08:40–08:49 UTC), 8 workers, OMP_NUM_THREADS=1.

## Setup

Common to every arm: gate `jp`, 20 `four_sat` Hamiltonians × 25 trials = 500 trials per cell, pure Gibbs cost
(λ1=0, λ=0, no β cap, no adaptive λ), SPSA (a = 0.2·√(37/n_params), c=0.15, A=10, η refresh every 5 steps),
`--seed 20260917` (the per-trial seed formula depends on final L and trial, so A/B/C/D share seeds per cell).
These match the existing baseline fleets exactly (`JP_STEP_SCAN_SUMMARY.md`).

| arm | encoding | init | SPSA steps (L=2 / 3 / 4) | source |
|---|---|---|---|---|
| A | binary | random L | 200 / 200 / 200 | `jp_L2_20260929T041311Z`, `jp_L3_20260929T041643Z`, `jp_B0_20260928T042134Z` (existing) |
| A400 / A800 | binary | random L | 400 or 800 | existing step scan (`jp_L{2,3,4}_s{400,800}_*`) |
| B | **gray** | random L | 200 / 200 / 200 | `jp_enc_B_gray_20260929T084122Z` |
| C | binary | **grow** from L=1, 200 steps **per stage** | **400 / 600 / 800** total | `jp_enc_C_bin_grow_ps200_20260929T084629Z` |
| D | **gray** | **grow**, 200 steps per stage | **400 / 600 / 800** total | `jp_enc_D_gray_grow_ps200_20260929T084923Z` |
| Ceq (aux) | binary | grow, equal-total 200 split over stages | 100+100 / 67+67+66 / 50×4 | `jp_enc_C_bin_grow_20260929T084222Z` |

Commands (`PYTHONPATH=. OMP_NUM_THREADS=1`):

```
python noiseless/run_u_sweep.py --u-names jp --layers 2,3,4 --trials 25 --steps 200 --workers 8 --seed 20260917 --lambda1 0 --optimizer spsa --encoding gray --tag jp_enc_B_gray
python noiseless/run_u_sweep.py ... --encoding binary --grow --grow-steps-per-stage 200 --tag jp_enc_C_bin_grow_ps200
python noiseless/run_u_sweep.py ... --encoding gray   --grow --grow-steps-per-stage 200 --tag jp_enc_D_gray_grow_ps200
python noiseless/run_u_sweep.py ... --encoding binary --grow --tag jp_enc_C_bin_grow   # aux, equal-total
python noiseless/analyze_encoding_grow.py   # -> encoding_grow_stats_summary.json + tables
```

Growth protocol: L=1 from `random_parameters(1, rng)` → SPSA → append last layer
[β_d, β_e = 0, θ_d = θ_e = π, φ_d = φ_e = 0] + N(0, 0.05²) kick on its 8 params (drawn from the trial rng) → fresh SPSA
(gain rescaled to 8L params, η controller restarted) → … up to final L. The equal-total plan (Ceq) was superseded by
Zach's generous per-stage budget; only the binary equal-total fleet had already run and is reported as auxiliary.

## Results at final depth

success = argmax bitstring is the GS; median / frac>0.5 over the 500 trials; best-of-25 = mean over H of max over 25 trials.
Wall = mean per-trial time (single process per trial).

| arm | L | total steps | success | mean p(GS) | median p(GS) | frac p(GS)>0.5 | best-of-25 | wall/trial (s) |
|---|---|---|---|---|---|---|---|---|
| A | 2 | 200 | 0.456 | 0.1297 | 0.0833 | 0.032 | 0.3061 | 0.30 |
| B (gray) | 2 | 200 | **0.538** | 0.1426 | 0.1119 | 0.032 | 0.3527 | 0.30 |
| C (grow) | 2 | 400 | 0.486 | **0.2051** | **0.1642** | **0.106** | 0.3385 | 0.47 |
| D (gray+grow) | 2 | 400 | 0.438 | 0.2010 | 0.1574 | 0.078 | 0.3399 | 0.45 |
| A400 | 2 | 400 | 0.486 | 0.1487 | 0.1058 | 0.048 | 0.3479 | 0.57 |
| A | 3 | 200 | **0.774** | 0.1581 | 0.1270 | 0.026 | 0.3646 | 0.42 |
| B (gray) | 3 | 200 | 0.762 | 0.1685 | 0.1304 | 0.032 | 0.3853 | 0.43 |
| C (grow) | 3 | 600 | 0.602 | **0.3052** | **0.2639** | 0.132 | **0.4948** | 0.89 |
| D (gray+grow) | 3 | 600 | 0.564 | 0.2883 | 0.2221 | **0.158** | 0.4915 | 0.87 |
| A400 | 3 | 400 | 0.804 | 0.1900 | 0.1516 | 0.050 | 0.4271 | 0.84 |
| A800 | 3 | 800 | 0.820 | 0.2207 | 0.1758 | 0.064 | 0.4844 | 1.66 |
| A | 4 | 200 | 0.932 | 0.1717 | 0.1434 | 0.018 | 0.3631 | 0.53 |
| B (gray) | 4 | 200 | 0.908 | 0.1784 | 0.1511 | 0.026 | 0.3548 | 0.56 |
| C (grow) | 4 | 800 | 0.752 | **0.3819** | **0.3864** | **0.354** | 0.5802 | 1.44 |
| D (gray+grow) | 4 | 800 | 0.762 | 0.3753 | 0.3538 | 0.288 | **0.6095** | 1.42 |
| A800 | 4 | 800 | **0.950** | 0.2558 | 0.2210 | 0.072 | 0.4863 | 2.16 |
| Ceq (aux) | 2 / 3 / 4 | 200 | 0.476 / 0.598 / 0.736 | 0.196 / 0.285 / 0.303 | 0.162 / 0.214 / 0.253 | 0.088 / 0.116 / 0.162 | 0.326 / 0.477 / 0.508 | 0.24 / 0.32 / 0.38 |

Matched-total-steps comparisons: C/D L=2 (400) ↔ A400 L=2; C/D L=4 (800) ↔ A800 L=4. **L=3 at 600 total steps has no
exact random-init match**; it is bracketed by A400 (0.804 / 0.190) and A800 (0.820 / 0.221).

### Per-stage trajectory (mean over 500 trials; 200 steps per stage)

| arm (final L) | L=1 succ / p(GS) | L=2 | L=3 | L=4 |
|---|---|---|---|---|
| C (4) | 0.146 / 0.105 | 0.546 / 0.223 | 0.656 / 0.320 | 0.752 / 0.382 |
| D (4) | 0.060 / 0.079 | 0.476 / 0.215 | 0.578 / 0.300 | 0.762 / 0.375 |

(Final-L=2 and 3 runs use different seeds, so their stage-1/2 numbers differ slightly; all in the stats JSON.)
Insertion check: p(GS) right after appending the transparent layer equals the previous stage's final p(GS) to **0.0**
(max |Δ| over all 9000 insertions in C, D and Ceq); the σ=0.05 kick changes mean p(GS) by −0.0056…+0.0008.

### Paired vs A (same Hamiltonian + trial + seed, n=500)

| comparison | L | frac p(GS) higher | mean Δp(GS) | median Δp(GS) | A-fail→succ | A-succ→fail |
|---|---|---|---|---|---|---|
| B vs A | 2 / 3 / 4 | 0.56 / 0.54 / 0.52 | +0.013 / +0.010 / +0.007 | ≈0 / ≈0 / ≈0 | 110 / 72 / 22 | 69 / 78 / 34 |
| C vs A | 2 / 3 / 4 | 0.63 / 0.67 / 0.75 | +0.075 / +0.147 / +0.210 | +0.041 / +0.109 / +0.245 | 90 / 49 / 21 | 75 / 135 / 111 |
| D vs A | 2 / 3 / 4 | 0.64 / 0.67 / 0.75 | +0.071 / +0.130 / +0.204 | +0.059 / +0.091 / +0.204 | 102 / 51 / 22 | 111 / 156 / 107 |
| C vs A400 (L2) / A800 (L4) | 2 / 4 | 0.60 / 0.68 | +0.056 / +0.126 | +0.010 / +0.161 | 82 / 15 | 82 / 114 |
| D vs C | 2 / 3 / 4 | 0.51 / 0.46 / 0.48 | −0.004 / −0.017 / −0.007 | ≈0 | 95 / 66 / 17 | 119 / 85 / 12 |

## Findings

- **Transparent layer (verified):** with this code's conventions R(π, φ=0) = exp(−iπσx/2) = −iσx and
  ECD(0) = σ⁻⊗I + σ⁺⊗I = σx⊗I, so ECD(0)·R(π,0) = −i·I exactly (global phase) and `jp` is Fock-diagonal. The planned
  setting β=0, θ=π, φ=0 is exactly probability-transparent (numerically 0.0 difference; unit test <1e-12 for jp, cz_nm,
  identity). It would NOT be transparent for a non-diagonal U (e.g. beamsplitters).
- **Layer growth is a big p(GS) lever but hurts success at L≥3.** At matched total steps, growth beats random-init on
  mean p(GS) (L2: 0.205 vs 0.149; L4: 0.382 vs 0.256), median (L4: 0.386 vs 0.221) and fraction of trials with
  p(GS)>0.5 (L4: 0.354 vs 0.072); best-of-25 at L4 0.58–0.61 vs 0.49. Even the equal-total 200-step growth (Ceq) beats
  800-step random init on mean p(GS) at L3/L4 (0.285/0.303).
  But success drops: L3 0.60 (C) vs 0.77–0.82, L4 0.75–0.76 vs 0.93–0.95. Growth outcomes are **bimodal**: grown
  circuits concentrate probability sharply, and when they lock onto the wrong basin the argmax is the **first-excited
  level** (100% of C failures at L3/L4, 94–98% for D) with p(GS) ≈ 0.02 in L4 failures (vs 0.08 for random-init
  failures). The L=1 stage succeeds only ~15% (binary) / ~6% (gray), and later stages largely refine the L=1 basin
  rather than escape it.
- **Gray encoding (B vs A): small, mixed effect.** L2 success +0.08 (0.538 vs 0.456) and slightly higher mean/median
  p(GS) at all depths, but L3/L4 success slightly lower (−0.012 / −0.024); paired medians ≈ 0, so mostly reshuffling
  which trials succeed. With growth, gray (D) ≈ binary (C) within noise (L4 success 0.762 vs 0.752; mean p(GS) slightly
  lower). Gray's L=1 stage is much worse (success 0.06 vs 0.15).
- **Gray changes what `jp` couples:** `jp` phases by physical Fock parity (n_A+n_B) mod 2. In binary, the Fock parity
  of a cavity is its LSB logical bit, so `jp` acts like a 2-body Z_{A,lsb}Z_{B,lsb} phase. In Gray, parity(n) =
  XOR of all 3 Gray bits, so `jp` becomes a 6-body Z_{A1}Z_{A2}Z_{A3}Z_{B1}Z_{B2}Z_{B3} phase in the logical basis.
  Gray does make single-photon displacement steps (n→n±1) flip exactly one logical bit.
- **Cost:** grown circuits spend early steps on cheaper shallow circuits: 800-step growth to L4 costs 1.44 s/trial vs
  2.16 s for 800-step random-init L4 (−33%).

## Wall times (8 workers, 1500 trials per fleet)

B 83 s; C (per-stage 200) 178 s; D (per-stage 200) 174 s; Ceq (aux) ≈ 60 s.

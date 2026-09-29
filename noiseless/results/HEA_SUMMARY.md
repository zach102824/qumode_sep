# HEA (hardware-efficient ansatz, lattice RY + CZ) — noiseless bake-off arm

Added 2026-09-29. New `hea` arm in `noiseless/run_ecd_vs_qaoa.py` (circuit in `noiseless/hea.py`,
tests in `noiseless/tests/test_hea.py`).

Source: `fleet_hea_20260929T073413Z` (full JSON gitignored; `_summary.json` / `_bestofn.json` committed).

## Circuit

8 qubits on a 2×4 square lattice, snake numbering:

```
q1 - q2 - q3 - q4
|    |    |    |
q8 - q7 - q6 - q5
```

- Start |0⟩^⊗8.
- Each layer l = 1..L: RY(θ) on all 8 qubits; then CZ sublayer A (1,2)(3,4)(5,6)(7,8),
  sublayer B (2,3)(4,5)(6,7), sublayer C rungs (1,8)(2,7)(3,6). 10 CZ per layer = the 10 lattice edges
  (rung 4–5 is covered by B's (4,5)).
- Final RY layer on all 8 qubits. RY only (no RZ); RY(θ) = exp(−iθY/2).
- n_params = 8(L+1): tiers 16 / 24 / 32 → L = 1 / 2 / 3. Layout `x.reshape(L+1, 8)`, row = RY layer, column k = q_{k+1}.

**Qubit mapping:** 1-indexed q_k ↔ repo qubit index k−1 ↔ bitstring character k−1 (MSB-first, basis index = int(bitstring, 2)),
i.e. q1 is the MSB and Z-term site 0 of the four_sat Hamiltonian (same convention as the QAOA arms' diagonal spectrum).

## Settings (identical to the QAOA-full arm except the circuit)

- Cost: Gibbs objective on the **full** four_sat diagonal spectrum, `SampledTailEta` η controller, same refresh cadence.
- SPSA: same `run_spsa`, c = 0.15, A = 10, a = `scale_spsa_a(n)` = 0.2·sqrt(37/n) → 0.3041 / 0.2483 / 0.2151.
- Init: **Uniform[0, π)** for every RY angle (same draw rule as `random_qaoa_params`).
- Scoring: true full 4-SAT GS; success = GS is the most-likely bitstring; p(GS) = Born probability of the GS.
- 25 trials × 200 steps × 20 Hamiltonians (N = 500 per tier), `--seed 20260917` (jp fleets' seed; bake-off used 20260918),
  8 workers. Total fleet wall ≈ 31 s for 1500 jobs.

Command:
```
python noiseless/run_ecd_vs_qaoa.py --arms hea --param-tiers 16,24,32 --trials 25 --steps 200 \
    --seed 20260917 --workers 8 --tag fleet_hea
```

## Results

| Params | L | Success | Mean p(GS) | Median p(GS) | Mean best-of-25 p(GS) | Min p(GS) | Wall / trial (mean) | N |
|-------:|--:|--------:|-----------:|-------------:|----------------------:|----------:|--------------------:|--:|
| 16 | 1 | 0.992 | 0.9335 | 0.9465 | 0.9849 | 0.0000 | 0.116 s | 500 |
| 24 | 2 | 1.000 | 0.8889 | 0.8962 | 0.9608 | 0.3383 | 0.156 s | 500 |
| 32 | 3 | 1.000 | 0.8344 | 0.8492 | 0.9364 | 0.3296 | 0.199 s | 500 |

Best-of-25 = per-H max over 25 trials, then mean over 20 H. The 4 failures at 16 params (000 t15, 003 t8, 003 t23, 004 t15)
converged to other product states with p(GS) ≈ 0; worst per-H success at 16 is 0.92. For reference the old bake-off's wall/trial was
ECD 0.27–0.49 s and QAOA-full 0.33–0.63 s (different run, 6 workers).

## Read / caveats

- HEA dominates every other arm on both success and p(GS) at all tiers (vs ECD `jp` 0.456–0.932 / 0.13–0.17).
- This is expected rather than suspicious: the 4-SAT GS is a computational basis state, i.e. a **product state**, and the final
  RY layer alone can prepare any basis state exactly (θ ∈ {0, π}). The Gibbs cost on a diagonal H rewards concentrating on
  low-energy bitstrings, which HEA can do with no entanglement at all — the landscape is close to a classical relaxation
  of the spin problem. The HEA is therefore a strong "trivially expressive for classical GS" baseline, not evidence of
  quantum advantage from entangling structure.
- p(GS) *decreases* with depth (0.93 → 0.89 → 0.83): more parameters with the same 200-step budget and a smaller SPSA gain,
  and random entangling layers that must be undone; success stays ~1.
- Seed differs from the original bake-off (20260917 vs 20260918).
- Existing arms verified unchanged: 6 trials re-run from `fleet_ecd_vs_qaoa_20260918T060037Z` (ecd, qaoa_full, qaoa_nn at
  16 params, H 000/010) reproduce p(GS) and final `x` bit-for-bit (`hea` was appended to `ARMS`, so seed arm codes are unchanged).

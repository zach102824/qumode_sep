# ECD speed study, n=20 3-cavity device (ks 6,6,5; nf 80/80/48; 3 transmons). Oct 14 2026, CST

Joint state: 2·2·2·80·80·48 = 2,457,600 amplitudes. All timings single-threaded (OMP/OPENBLAS/MKL = 1) unless noted, at most 6 cores in use.
Code: `noiseless/ecd_bench.py` (benchmark + fast circuit), `noiseless/ecd_profile.py`, `noiseless/ecd_validate.py`, `noiseless/ecd_trunc.py`.
Flag: `run_relayout_kbit.py --ecd-fast` (sets QM_ECD_FAST=1; `XorMultiCavSim.probs_from_x` then uses `apply_circuit_multi_fast2`). Default unchanged.
Raw outputs: `noiseless/results/ecd_speed/`.

## How the current code does it
`relayout_3cav.apply_circuit_multi`: per layer, 3 transmon rotations (tensordot + moveaxis), then for each pair an ECD:
D(β/2) built with the eigen trick already (`ecd_kbit.displace_fast`: eigh of the generator cached once per nf, each β = diagonal phase + two 80×80 basis changes, 0.0902 ms per D),
applied with take / tensordot / moveaxis / stack (several full-state copies), then the jp phase. jp is diagonal: values exactly 1, -1, -1j (one elementwise multiply, 11.76 ms per layer).
Optimizer is SPSA/Adam on function values only (no gradients), so only the forward pass matters.

## Step 0: profile (shortened n=20 trial, instrumented, 35.80 s window, old code)
| part | s | share |
|---|---|---|
| ECD apply (take/tensordot/moveaxis/stack) | 18.369 | 51.32% |
| ECD build D(β) | 0.080 | 0.22% |
| transmon rotations | 6.102 | 17.05% |
| jp phase | 0.959 | 2.68% |
| readout: born | 0.497 | 1.39% |
| readout: Gibbs cost | 1.172 | 3.27% |
| everything else (SPSA bookkeeping, classical search, process start) | 8.616 | 24.07% |

## Step 1: one ECD layer (3 ECDs) on a realistic n=20 state (2 random layers of the real circuit), βs 2.1+1.3j, -1.7+0.4j, 0.9-2.2j
| method | ms per layer | max abs err vs current | speedup |
|---|---|---|---|
| (a) current: eig D + tensordot/moveaxis/stack | 150.53 | 0 | 1.00× |
| (b) closed-form Laguerre D (truncated), batched matmul(out=) | 71.02 | 3.03e-16 * | 2.12× |
| (c) expm_multiply on the state (sparse generator) | 4350.09 | 2.90e-16 | 0.0346× |
| (d1) eig D + batched matmul(out=), ping-pong buffers | 71.57 | 0 | 2.10× |
| (d2) eig trick on the state (two basis changes, no D built) | 236.47 | 1.08e-16 | 0.637× |
| (e) d1 in complex64 | 1388.73 | 2.68e-08 (fails 1e-10) | 0.108× |
| (f) split e^{-a²/2} exp(α a†) exp(-α* a) | 75.11 | 3.28e-16 * | 2.00× |
| (g) scipy expm D + batched matmul | 77.70 | 3.07e-16 | 1.94× |
\* (b),(f) are the infinite-space D cut to 80 levels, not unitary; they match here only because the state has no weight near the cutoff (see check 2). (e) is slow because numpy has no strided cgemm path here; it also fails accuracy.
D build cost alone (nf 80): eig 0.0902 ms, Laguerre 0.961 ms, scipy expm 1.028 ms.
The gain is all in how D is applied (BLAS gemm into preallocated buffers, no copies), not in how D is built.
(d1) runs at about 57 GFLOP/s on one core (compute bound), so further ECD gains need fewer flops, not a different D formula.

## Extra ideas, full L=4 circuit (rotations + ECD + jp), state from vacuum
| version | ms per circuit (1 thread) | ms (6 threads) | max abs err |
|---|---|---|---|
| old `apply_circuit_multi` | 864.75 | 694.33 | 0 |
| fast: d1 ECD + in-place rotations | 390.13 (2.22×) | 525.30 | 6.07e-18 (probs) |
| fast2: + layer 1 built analytically as a product of 3 pair states (start is vacuum), 3 rotations fused into one 8×8 Kronecker matmul, final jp skipped (phase, invisible to born) | 233.89 (3.70×) | 305.09 | 6.07e-18 (probs) |
Threads do not help (memory bound at this size); best use of 6 cores is 6 single-thread trials in parallel. numba was not tried: the ECD is already a near-peak BLAS gemm.

## Step 2: end-to-end, one full n=20 sl3 trial (inst 0, same seed), run side by side
| | wall per trial | success | p(GS) | evals | lookups |
|---|---|---|---|---|---|
| old | 1281.4 s | True | 0.954885746164735 | 2005 | 600 |
| --ecd-fast | 504.2 s | True | 0.954885746164735 | 2005 | 600 |
Speedup 2.541×; p(GS) identical to all printed digits (diff 0), and equal to the logged sl3 value.
10-trial screen estimate with 6 workers (2 waves): old 1687 s per trial at 8 parallel → new about 664 s per trial, about 1328 s (22 min) per screen, vs about 3374 s (56 min) for the old code on 8 workers (estimate, not measured).

## Validation
(1) Small copy, 3 cavities nf 12/12/8 + 3 transmons (N = 9216), dense operators from scipy expm of each generator in controlled-displacement block form:
one-layer joint ECD, fast method on every basis vector vs dense: max abs 1.129e-15 (dense ECD unitarity error 1.554e-15).
Full L=4 circuit (rotations, ECD, jp) as a dense matrix vs fast method on every basis vector: max abs 1.824e-15.
L=4 state from vacuum: old 1.387e-15, fast 1.374e-15; fast2 probabilities 1.735e-16.
(2) nf=80 D vs analytic Laguerre formula (mpmath, 40 digits), |α| = |β|/2:
| abs α | max err, all 80×80 | m,n below 60 | m,n below 40 |
|---|---|---|---|
| 0.25 | 0.429 | 3.00e-15 | 1.73e-15 |
| 0.5 | 0.338 | 2.31e-15 | 2.30e-15 |
| 1.0 | 0.269 | 6.92e-14 | 2.07e-15 |
| 1.5 | 0.239 | 2.03e-07 | 2.70e-15 |
| 2.0 | 0.222 | 0.00138 | 3.52e-15 |
| 2.5 | 0.212 | 0.128 | 3.06e-15 |
The big errors sit only near the cutoff, where the unitary truncated expm (our model) and the infinite-space D differ by construction. Final n=20 |β| is at most 0.9957, so |α| at most 0.498; explore inits go up to |β| 3 (|α| 1.5).
(3) Truncation, the 10 sl3 n=20 final parameter sets: probability in the top 5 Fock levels of each cavity is at most 8.41e-32 (all trials, all cavities).
Raising cutoffs 80/80/48 → 96/96/64 changes p(GS) by at most 4.44e-16. 9/10 reproduce the logged p(GS) to 4.4e-16. Inst 4's selected result is from an explore round whose layout is not stored in x/xa, so its p(GS) could not be rebuilt (its truncation numbers still hold).

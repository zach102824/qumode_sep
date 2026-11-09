# Correction to the GR query ledger and the Nov3 median

The original run files and `SUMMARY.md` tables are unchanged. This note corrects the accounting in the written report. No new search was run. Validation and final seeds were not opened. There is no hardware claim.

## 1. What “1024” was, and what was actually evaluated

The SPSA counter is an optimizer cap. It is not the number of energy evaluations the run performed.

Inside `run_spsa`, every call is one energy evaluation and the counter checks that total against the cap:

| cap | runs | calibration | iteration | candidates | SPSA counter |
|---:|---:|---:|---:|---:|---:|
| 128 | 1440 | 32 | 88 | 8 | 128 |
| 1024 | 1440 | 32 | 984 | 8 | 1024 |

Sum of those counters: 1,658,880. Monitoring inside the cap: 0. Counter mismatches in the 2880 files: 0.

Before that loop, `run_one` calls the same energy function once on each of the 4 initial parameter vectors. All 2880 stored runs have that record. The best of those four energies is start 0 in 564 runs, start 1 in 872, start 2 in 722, and start 3 in 722, so the calls are not a single unused placeholder. The stored field `counted_query: false` means “not inside the SPSA counter.” The calls were performed. They do not replace the SPSA-selected point; the reported optimum is still the best SPSA candidate by energy.

Inclusive energy-evaluation ledger for the pilot search:

| item | energy evaluations |
|---|---:|
| SPSA cap | 1,658,880 |
| 4 initial-state energy evaluations × 2880 runs | 11,520 |
| full search cost | 1,670,400 |

A cap-1024 run is 1028 energy evaluations. A cap-128 run is 132. The report’s “strict 1024 inclusive” sentence was wrong for the full cost.

Fidelity is not another query. `evaluate_nov9`, `evaluate_nov3`, and `evaluate_qaoa` build one state and return both the energy and the overlap. The overlap is a contraction with the stored ground state. It does not call the energy function again, and it is not used to choose the SPSA point.

Separate from the search, and also energy evaluations with fidelity attached to the same state:

- Cutoff prestudy, counted from the script and `cutoff.json`: 192 ladder cells × 10 evaluations, plus 96 leak-probe points × 2 evaluations, plus 7 timing evaluations = 2,119. The probe stopped at `c = 0.1` (`alpha_probe_worst_dp` ≈ 8.8×10⁻⁴).
- Post-selection NF = 48 check: 2,112 true-displacement selected points × 3 evaluations = 6,336. All 2,112 passed. These calls are not in the 1,670,400.

The conformance suite is a test, not part of the search ledger.

## 2. One reconstructed N = 9 true-displacement state

File `runs/n9_s10000_b10_L1_inter_id_c1024.json`. Seed 10000, N = 9, b = 1, L = 1, interleaved binary, Uz, NF = 24. The SPSA counter in the file is 1024 (32 + 984 + 8).

An inline propagator and an observable routine that does not call `model._observe` reproduce the stored numbers:

| quantity | stored | independent |
|---|---:|---:|
| energy | −9.005640627738705 | −9.005640627738705 |
| fidelity | 0.7351219388482645 | 0.7351219388482637 |
| leak probability | 3.978×10⁻⁴ | 3.978×10⁻⁴ |

The plus state on this instance has energy −9 and fidelity 0.73347. The selected state is 0.00564 below that energy and 0.00165 above that fidelity. The plus stall on this saved point is the state that was selected, not a summary artifact. Re-evaluating the saved parameter vector with the production energy function matches the independent numbers to 10⁻¹⁵.

## 3. Why this Nov3 median is 0.9727 and the parent’s local figure is 0.9532

Both numbers are for Nov3rd, N = 9, b = 1, L = 1, cap 1024, seeds 10000–10007. They are not the same sample, and this repository does not contain the parent’s local source, so the difference is not labeled a bug in either direction.

Re-evaluating the eight saved parameter vectors reproduces the stored fidelities to 2×10⁻¹⁵. Sorted fidelities:

0.94053, 0.95480, 0.96581, 0.97188, 0.97348, 0.98104, 0.98441, 0.99075.

The median used in the report is the mean of the two middle values, 0.97188 and 0.97348, which is 0.97268. The lower of those two is already 0.9719. No choice among min, lower median, or upper median of this list equals 0.9532.

Per seed, fidelity, plus overlap, and the sha256 of the float64 little-endian upper triangle of J:

| seed | F | F+ | J upper sha256 |
|---:|---:|---:|---|
| 10000 | 0.97348 | 0.73347 | `31f3d9f3a33cf52744550c27d7ed487a6c01e2678b9354336dd0fa6cd81fb89c` |
| 10001 | 0.98104 | 0.74974 | `82e3a3a12a7ba6862ab46cb26233a2f646725c81439fce7eb4a077aa81b37f91` |
| 10002 | 0.96581 | 0.72189 | `3484d60b6ea91b15ad7d197a380b383372a9bdf120c1e99f198f1f516a4ef23b` |
| 10003 | 0.95480 | 0.70031 | `495147145c2fc6898820bfe9a3eea411e908161f8db820c4186759564e9c6e54` |
| 10004 | 0.94053 | 0.65687 | `c61b325c460f3ac509df3d32e4e0602f0d53434e1fac5ef2b640bb522dbd519c` |
| 10005 | 0.99075 | 0.85455 | `1e05cbcbde93e708fd45932f36538e1dd5369cb0bcc9a3c1ed7fd72e339739a6` |
| 10006 | 0.98441 | 0.78788 | `a51f0403559659cf19dacb82af9cc3cb98dd0d15ecd456eceaa24fdc0c6a6785` |
| 10007 | 0.97188 | 0.73910 | `6b1271d30fe099e523b8d1c74aac8d5b8243488e9d3340417d6b8ed20aad6a39` |

The median F+ of these eight instances is 0.7363. That matches the consultation’s quoted local median F+ of 0.736 at N = 9, b = 1. The disorder sample is consistent with that quote. The optimized Nov3 fidelities are a separate object.

Controls that this run actually used, and that are not fixed to one trajectory by the consultation text:

- Start E is γ = 0.25, θ_K = θ_M = 0, shell angles 0. The three random starts draw γ and each θ from U(−0.5, 0.5) and each shell angle from U(−0.3, 0.3), with `SeedSequence([seed, N, round(10b), L, method_id=1, start_id, stream])`. `PROTOCOL.md` does not specify the Nov3 start vector. `PREREG.md` does.
- The shell coordinates are the protocol polynomial `s1 p + s2 C(p,2) + s3 [p=3]`, with σ_s = 0.15 in those coordinates. The Nov 8 spin-glass code optimizes the Walsh coefficients `t1, t2, t3`. Those charts give the same gates up to a global phase, and they do not give the same SPSA step. This repository does not show which chart produced 0.9532.
- SPSA uses c = 0.1, A = 25, and one global median of `|ĝ_j|/σ_j` over the four calibration gradients to set the initial step to 0.5σ. A different pooling of that median changes the gain. For seed 10000 the four per-start gains `a` are 0.444, 0.275, 0.090, and 0.082. The selected point is start 1, last iterate, not the Polyak average.
- Iteration queries are 984 and monitoring queries are 0. `PROTOCOL.md` §5 says the local audit spent 340 of 1024 queries on monitoring. A figure from that monitored budget is a different optimizer.
- Selection is the lower energy of the last iterate and the mean of the last 64 post-update vectors, then the best of 4 starts. The protocol table says 4 starts. A later sentence still says “the better of the 2 starts.” This run followed the table.

What does match the written Nov3rd definition: one shared γ, KM features with the RMS hat, interleaved packing `[a, a+N/3, a+2N/3]`, binary Fock 0..7, ideal `W = H^{⊗3}`, and the rewrite `|+⟩ → [UcZ → WSW]^L`. The older Nov_3rd plan, with a per-layer QAOA warm start and `φ_j = −β j`, is a different ansatz. Which of those the local 0.9532 used is not in this tree.

The two medians should be compared only after the J hashes above, the shell coordinate chart, the start vector, c, and the monitoring count are the same. They are not established to be the same here.

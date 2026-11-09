# LEXACT M2 — further speedup at N15/N18, not a GR adoption

M1 stands. This axis is not another gain rescue.

A numpy ping-pong SIMD FWHT was slower than M1 at N18 (14 ms vs 3.1 ms per transform) because it writes a second buffer every stage. That path was not adopted. The preregistered candidate keeps M1’s in-place butterflies, runs disjoint pairs at frozen T=4, and applies diagonal phases as `cis` from real angles instead of `numpy.exp` of a stored complex vector. T was not scanned. BLAS=1. Latency timings used one process.

## Result

At the N18 bottleneck the eval is faster than M1: **Nov3 L1 25.83 ms → 7.30 ms (3.54×)**, QAOA L1 12.74 ms → 3.81 ms (3.34×). Versus the original reference that is about 12×. N15 L1 Nov3 is 2.23 → 0.91 ms (2.45× vs M1).

It is **slower than M1 at N6 and N9** (thread overhead). N12 L6 Nov3 is 0.88× vs M1. The axis is a scaling win, not a uniform win. Resources: 4 kernel threads, 1 process, RSS 273 MB.

524 reference checks passed. SPSA traces of 768 new queries match to ≤1e-13 and pick the same start/candidate. F is post hoc. N3COND 772224 is unchanged.

## Stop rule

The N18 L1 Nov3 gate vs M1 passed, so the axis is kept as a versioned candidate. It is still not a GR adoption. No thread search. No N18 L>1 panel in this batch.

A 1000-instance N18 grid is still many hours, now with ~7 ms/eval at L=1 on 4 cores of one machine rather than ~23 ms on one core. Next in-scope cost is **many independent instances**, which is process-level parallelism of this kernel, not another transform rewrite and not SPSA conditioning.

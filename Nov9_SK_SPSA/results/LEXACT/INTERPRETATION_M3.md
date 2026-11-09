# LEXACT M3 — stop

This is process-level parallelism of the existing kernel, not another FWHT. K=4 and T=1 were frozen. T=4 inside each worker was not used.

On this 4-core VM the N18 L1 Q=128 pair (Nov3+QAOA) runs at **2413 instances/hour** vs 815 serial, 2.96× wall, bit-identical traces. Memory about 1 GB with four workers. Field prep remains ~45 s of classical Lanczos per new N18 B* instance and is not Q.

A 1000-instance N18 grid at this short-trace size would still be tens of minutes of SPSA plus hours of B* if every seed is new. Full Q=qL L=6 is out of this batch. Candidate only. GR unchanged. Conditioning stays closed. No further axis in this report.

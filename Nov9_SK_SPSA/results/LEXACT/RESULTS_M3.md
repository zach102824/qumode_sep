# LEXACT M3 measured instance-parallel throughput

Frozen: spawn Pool **K=4**, BLAS=1, kernel **T=1** (no M2 T=4 oversubscribe). Queue: static task list, `imap_unordered`. No new transform. Pid 11304, compile 0.122 s. N18 L=1, seeds 10032–10035, all accepted B* (odd E1). Field prep of the three new instances was 42–48 s each (133 s wall in the first attempt); this timing run reused them (0.05 s).

## Wall throughput

| batch | serial T=1 | K=4 pool | speedup |
|---|---|---|---|
| 4 Nov3 energy spots (plus ref/F) | 2.54 s | 0.85 s | 2.98× |
| 4 instances × 2 circuits × Q=128 SPSA | 17.67 s | **5.97 s** | **2.96×** |

Q=128 SPSA, 8 fits, 1024 queries, 0 outside cap:

- **2413 instances/hour** (an instance = one seed, Nov3+QAOA)
- **4826 circuit-fits/hour**
- serial: 815 instances/hour, 1630 fits/hour

Ideal 4× on 8 jobs would be ~4.4 s; 5.97 s is about 74% of that, spawn and two waves of four. Parent RSS 223 MB. Worker RSS 184–206 MB each. Four workers + parent ≈ 1.0 GB envelope.

## Conformance

Energy: pool vs serial dE=0, dstate0=0; vs reference |ΔE|≤5e-14, |ΔF|≤8e-15. SPSA: same start, same candidate, trace |ΔE|=0 on all 8 fits. Worker `numba_threads=1`. N9 spawn tests (9/9) passed before this batch.

N3COND 772224 unchanged. LEXACT SPSA 1536+1024=2560. GR recipe not patched. Not a 1000-seed panel. Stop.

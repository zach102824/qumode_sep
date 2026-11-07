# Round 4 V

Independent replay of the sublattice + block-mean family, then fresh QAOA and Rfull searches.

`sim.py` is a new logical circuit and a new physical packing circuit. It reads `source_data/` (manifest, packings, stored angles, stored states). It does not import the other agent's simulator.

```bash
cd Nov_3rd/tfim_shared_gamma/round4_V
OMP_NUM_THREADS=1 python3 test_replay.py
OMP_NUM_THREADS=1 python3 run_round4.py
```

`test_replay.py` checks the block-mean projection, physical versus logical overlap, joint site relabeling, analytic gradients, and the L5/L6 replay. `run_round4.py` repeats that replay and then runs the pre-registered searches. Seeds and eval caps are constants at the top of `run_round4.py`.

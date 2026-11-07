# Round 6 RP

Independent replay of the stride-4 states from PR73 commit `d9af7f2a`. The simulator is this branch's `round4_V/sim.py`. The other branch's code is not imported.

```bash
cd Nov_3rd/tfim_shared_gamma/round6_RP
OMP_NUM_THREADS=1 python3 replay.py
```

`source_data/` holds the imported cells, state archives, packing, and the raw `nfev` event list. `results/baseline_v1.json` is the versioned baseline.

# Nov3 initialization and restart allocation

Development only, on original Nov3 (`|+⟩ → [UcZ → WSW]^L`) with shared `γ`, `θ_K`, `θ_M` and three shell parameters per layer. QAOA `p = L` gets the same query budget and the same allocation. SPSA only. Energy selects. Fidelity is post hoc.

Seeds `10018..10025`. Train half `10018..10021` freezes the arm. Test half `10022..10025` is confirmatory. Validation and final seeds are refused. `B*` is the interior entropy maximum on sites `0 .. floor(N/2)-1`.

v1 compares `equal4` (frozen matched-recipe gains and boxes, budget `Q = q L`) with `halve8to2` (correlated cold starts, eight energy screens, two SPSA finalists) at `N = 9,12`, `L = 1,3`, `q = 128,512,1024`.

```bash
python3 -m Nov3_INIT.conformance
python3 -m Nov3_INIT.run_v1 --phase bstar
python3 -m Nov3_INIT.run_v1 --phase train
python3 -m Nov3_INIT.run_v1 --phase freeze
python3 -m Nov3_INIT.run_v1 --phase test
python3 -m Nov3_INIT.run_v1 --phase summary
```

This does not retune the frozen 1000-seed benchmark and it does not run Nov9 displacements.

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

v2 (`PREREG_V2.md`) changes only the Nov3 deterministic center and reuses the v1 QAOA ledger.

```bash
python3 -m Nov3_INIT.run_v2 --phase train
python3 -m Nov3_INIT.run_v2 --phase freeze
python3 -m Nov3_INIT.run_v2 --phase test
python3 -m Nov3_INIT.run_v2 --phase summary
```

v4 (`PREREG_V4.md`) is a later development fit, not a fresh holdout. It keeps `center4` as a candidate and gives QAOA the derived center `(γ, β) = (0.25, 0.25)` on every layer. The grid is `N = 9,12` at `L = 4,6` and, after the sparse probe, `N = 15` at `L = 1`, seeds `10018..10025`, `q = 512,1024`.

```bash
python3 -m Nov3_INIT.conformance_v4
python3 -m Nov3_INIT.run_v4 --phase n15
python3 -m Nov3_INIT.run_v4 --phase run
python3 -m Nov3_INIT.run_v4 --phase summary
```

v5 (`PREREG_V5.md`) keeps those exact four-start recipes and asks whether the N=15 L1 energy edge persists at N=18. Resource gate first. Seeds `10018..10025` only.

```bash
python3 -m Nov3_INIT.conformance_v5
python3 -m Nov3_INIT.run_v5 --phase bstar
python3 -m Nov3_INIT.run_v5 --phase run
python3 -m Nov3_INIT.run_v5 --phase summary
```

v6 (`PREREG_V6.md`) keeps those exact recipes at N=18, L=2 and L=3. It reuses the v5 B* files. Resource gate first.

```bash
python3 -m Nov3_INIT.conformance_v6
python3 -m Nov3_INIT.run_v6 --phase run
python3 -m Nov3_INIT.run_v6 --phase summary
```

v7 (`PREREG_V7.md`) keeps those exact recipes at N=15, L=2 and L=3. It reuses the v4 B* files. Not LE S1 and not local R10.

```bash
python3 -m Nov3_INIT.conformance_v7
python3 -m Nov3_INIT.run_v7 --phase run
python3 -m Nov3_INIT.run_v7 --phase summary
```

This does not retune the frozen 1000-seed benchmark and it does not run Nov9 displacements.

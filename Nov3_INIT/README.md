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

v8 (`PREREG_V8.md`) is a fresh L=2 holdout on unread seeds `10144..10151` at N=15 and N=18. Same v4 recipe. Gates H1/H2 at Q=2048. Does not patch GR.

```bash
python3 -m Nov3_INIT.conformance_v8
python3 -m Nov3_INIT.run_v8 --phase bstar
python3 -m Nov3_INIT.run_v8 --phase run
python3 -m Nov3_INIT.run_v8 --phase summary
```

v9 (`PREREG_V9.md`) is an n=16 N=15 L=2 Q=2048 effect-size panel on unread seeds `10200..10215`. Estimation only. No GR patch.

```bash
python3 -m Nov3_INIT.conformance_v9
python3 -m Nov3_INIT.run_v9 --phase bstar
python3 -m Nov3_INIT.run_v9 --phase run
python3 -m Nov3_INIT.run_v9 --phase summary
```

v10 (`PREREG_V10.md`) is an n=16 N=18 L=2 Q=2048 effect-size panel on unread seeds `10232..10247`. Estimation only. No GR patch.

```bash
python3 -m Nov3_INIT.conformance_v10
python3 -m Nov3_INIT.run_v10 --phase bstar
python3 -m Nov3_INIT.run_v10 --phase run
python3 -m Nov3_INIT.run_v10 --phase summary
```

Pair-shell (`PREREG_PAIR.md`) unshares the existing 3-shell SNAP per cavity. That probe did **not** implement paper `V(M)`. N=15, L=2 and L=3, seeds `10328..10335`, same `Q=1024·L` on shared Nov3, pair-cavity, and QAOA. No GR patch.

```bash
python3 -m Nov3_INIT.conformance_pair
python3 -m Nov3_INIT.run_pair --phase bstar
python3 -m Nov3_INIT.run_pair --phase run
python3 -m Nov3_INIT.run_pair --phase summary
```

Shell-gain (`PREREG_SHELLGAIN.md`) keeps shared-Uc Nov3 and scales only shell `a_k` ×2 or ×4 vs Uc. N=15 L=3, seeds `10368..10375` (**shared-with-LE-PR101**, not exclusive), `Q=3072`. See `ADDENDUM_SHELLGAIN.md`. No GR patch.

```bash
python3 -m Nov3_INIT.conformance_shellgain
python3 -m Nov3_INIT.run_shellgain --phase bstar
python3 -m Nov3_INIT.run_shellgain --phase run
python3 -m Nov3_INIT.run_shellgain --phase summary
```

Spread8 L=3 (`PREREG_SPREAD8.md`) compares eight short SPSA blocks to four long ones on original shared-Uc Nov3 vs QAOA. N=15 L=3, seeds `10376..10383`, `Q=3072`. No GR patch.

```bash
python3 -m Nov3_INIT.conformance_spread8
python3 -m Nov3_INIT.run_spread8_l3 --phase bstar
python3 -m Nov3_INIT.run_spread8_l3 --phase run
python3 -m Nov3_INIT.run_spread8_l3 --phase summary
```

Anneal L=3 (`PREREG_ANNEAL.md`) compares v4 Spall `0.602/0.101` to asymptotically optimal `α=1`, `γ=1/6`. N=15 L=3, seeds `10456..10463`, `Q=3072`. No GR patch. No re-fit of `10368..10375`.

```bash
python3 -m Nov3_INIT.conformance_anneal
python3 -m Nov3_INIT.run_anneal --phase bstar
python3 -m Nov3_INIT.run_anneal --phase run
python3 -m Nov3_INIT.run_anneal --phase summary
```

Paper \(\mathcal{V}(M)\) (`PREREG_VM.md`) is a **separate architecture arm** (arxiv 2204.02923): independent all-to-all ZZ pair-phase plus shared Rx mixer at \(p=L\). Not Nov3 shells and not a Nov3 retune. N=15, L=2 and L=3, seeds `10480..10487`, same `Q=1024·L` on shared-shell Nov3, paper V(M), and QAOA. No GR patch.

```bash
python3 -m Nov3_INIT.conformance_vm
python3 -m Nov3_INIT.run_vm --phase bstar
python3 -m Nov3_INIT.run_vm --phase run
python3 -m Nov3_INIT.run_vm --phase summary
```

Fock-cutoff (`PREREG_CUTOFF.md`) is leakage validation of frozen v4 center-4, not an optimizer retune. D0 = NF=8 (Hilbert dim 8 per cavity). D0+2 = NF=10 with ideal \(W_L\) identity pad (algebraic invariance, not a SNAP–D pulse-compile cutoff test). N=15, L=1 and L=2, seeds `10520..10527`. Compiled \(W_L\) is not modelled. No GR patch. See `results/cutoff/ADDENDUM_AUDIT.md`.

```bash
python3 -m Nov3_INIT.conformance_cutoff
python3 -m Nov3_INIT.run_cutoff --phase bstar
python3 -m Nov3_INIT.run_cutoff --phase run
python3 -m Nov3_INIT.run_cutoff --phase summary
```

Non-ideal \(W_L\) (`PREREG_WL.md`, `AUDIT_WL.md`) is a **phenomenological** \(G_{7,8}(\varepsilon=0.10)\) sensitivity test, not a compiled-\(W_L\) or hardware-calibrated model. PROTOCOL §3.1 names the 7–8 displacement edge; it does not specify this Givens or \(\varepsilon\). N=15 L=1 Q=1024, seeds `10528..10535`. Secondary B−A is Nov3-only; equal \(p_{\rm leak}\) is not equal energy bias. SNAP–D pulse compile of \(W_L\) remains unmodelled. No GR patch. No further \(\varepsilon\) sweeps.

```bash
python3 -m Nov3_INIT.conformance_wl
python3 -m Nov3_INIT.run_wl --phase bstar
python3 -m Nov3_INIT.run_wl --phase run
python3 -m Nov3_INIT.run_wl --phase summary
python3 -m Nov3_INIT.audit_wl
```

Native \(W_{\rm BinH}\) compile (`PREREG_WCOMP.md`) is a **compiler feasibility** task, not an SK seed panel. SNAP + true real-\(\alpha\) displacement, \(K=3\), SPSA \(Q=2048\) compiler evaluations. SK ledger unchanged. Not flat prep, not \(U_X\), not cropped \(D_8\). No GR. Not a hardware claim.

```bash
python3 -m Nov3_INIT.conformance_wcompile
python3 -m Nov3_INIT.run_wcompile --phase fit
python3 -m Nov3_INIT.audit_wcompile_prop
```

Corrected-workspace compile (`PREREG_WCOMP_WS.md`) is one bounded simulator correction of that trial: common Fock workspace 96, frozen replay 128, no per-gate \(P_{16}\). Same \(K=3\), real \(\alpha\), SNAP 0..7, \(Q=2048\). Coherent isometry \(L\) selects; identity is the measured incumbent. PR109 remains history (projection-exploitable objective). SK ledger 2818048. No extra \(K\)/\(\alpha\) after a miss. Not a hardware claim.

```bash
python3 -m Nov3_INIT.conformance_wcompile_ws
python3 -m Nov3_INIT.run_wcompile_ws --phase fit
```

SK spin-flip gauge (`PREREG_GAUGE.md`) is a bounded labeling-robustness panel of the original Nov3 L=1 edge, not a compiler or ansatz retune. N=15 L=1 Q=1024, seeds `10552..10559`, identity + three frozen random gauges. Statistical unit is the 8 instances. Exploratory; no adoption; no GR. Compiler stream closed.

```bash
python3 -m Nov3_INIT.conformance_gauge
python3 -m Nov3_INIT.run_gauge --phase bstar
python3 -m Nov3_INIT.run_gauge --phase check
python3 -m Nov3_INIT.run_gauge --phase run
python3 -m Nov3_INIT.run_gauge --phase summary
```

Unread-seed L=1 width (`PREREG_WIDTH.md`) asks whether the shallow Nov3−QAOA edge appears on exclusive instances at **both** N=15 (`10568..10575`) and N=18 (`10576..10583`). Frozen v4 center-4, Q=1024, L=1 only. Not L=2 sampling. Not pooled across N. No GR.

```bash
python3 -m Nov3_INIT.conformance_width
python3 -m Nov3_INIT.run_width --phase bstar
python3 -m Nov3_INIT.run_width --phase run
python3 -m Nov3_INIT.run_width --phase summary
```

This does not retune the frozen 1000-seed benchmark and it does not run Nov9 displacements.

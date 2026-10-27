# Diagnostics to run later (not blocking densify)

From Opus round-2 experiment 3 — do **not** block the train queue on these.

## 1. Existing-weights forward-pass
- Compare `g_tied`, `w_tied`, per-layer |α| vs `amax`=1.8 for champion vs seeds 2/3/4 and collapsed runs (leak Med 300–600).
- Plateau signature: cost effectively switched off.
- Trigger for later `g0` sweep: champ gamma far from 3.0 while plateau stays near 3.0 or →0.

## 2. Untied checkpoint checks
- Source: `results/oct23_exp/gnn5_n8_pure_cw_theta.npz`
- Per-layer gamma variation; replace each layer gamma/`w` with layer mean, eval Med (no train).
- If projected Med ≲60 or gamma nearly flat → worth tied homotopy. If ≳150 or strong ramp → drop.

## 3. Per-instance tied oracle
- ~30 test formulas; optimise L=5 circuit (A, TH, g, w; ~200 params) with ~16 restarts.
- If oracle Med ≲8 → gap is optimisation (push multi-start/anneal).
- If oracle Med ~20–30 → 14 unreachable at tied L=5 → move to L=7/9 with screen protocol.

## 4. Larger test set
- Build ~1000-formula test with WalkSATlm reference before claiming Med ≲30 (current test n=100 has ~±13% Med noise).

# beat_qaoa_12h STATUS

**Updated:** 2026-11-03 ~20:47 Asia/Shanghai  
**Phase:** Round 0 nearly done; Round 1 L=2 trio training

## Round 0
- Hash disjoint: **PASS** (n6/9/12 sealed∩scale=0; n15 sealed train empty → generated beat packs)
- Dual L=1: **PASS** R=1.000000 all CORE
- Dual L=2: **PASS** R=1.000000 all CORE (after paper-n6 retrain + ladder refresh)
- Dual L=16: running (QAOA ladder force-refreshed; stale legacy n9 Med 2.24→1.58)

### Fixes applied
- Archived non-paper L2 QAOA → `qaoa_angles/L2_legacy_nonpaper`
- Force-refreshed L2 + L16 QAOA scale1000 ladders (angles updated earlier without re-eval)
- Patched `ensure_qaoa.py` so retrain forces full CORE re-eval

## Round 1 (running)
- Idea1 CORE-RATIO L=2 seed0
- Idea2 RULE-SCALED L=2 seed0 κ0=1
- Idea3 TRUST-RESIDUAL L=2 seed0 ρ=0.1
- Idea1 L=16 seed0: queued after L16 dual PASS

Train packs: `train_packs/train_n{6,9,12,15}.npz` N=250 each, disjoint from scale1000.

# explore_OCT29

OCT 29 **small→large transfer** track for coherent multi-cavity SAT.

**Step 0 implemented** (Zach go 2026-10-29): shared SNAP+D + `γ_ℓ`, Φ = `H_cost`.

```text
|0⟩ → D(α₀) → [SNAP → D → U_Φ]^L
U_Φ = exp(-i γ_ℓ Φ),  Φ = H_cost
```

| Doc | What |
|-----|------|
| [`PLAN.md`](PLAN.md) | Theory lock, freedom ladder, promote gate, so-far bars |
| [`RUN.md`](RUN.md) | Box commands (I run these; you confirm plan in chat) |
| [`TODO_REWRITE.md`](TODO_REWRITE.md) | Step 0 checklist (done) + later ladder |

## Locus

- **Box CPU** — this track (nonstop evolve).
- **Chaodao GPU** — deferred.
- Git: `/workspace/qumode_sep/explore_OCT29`

## Train / eval / promote

- Train: **n = 6 & 9**
- Promote gate: **zero-shot n12** Med/B90 (train-n alone never wins)
- Spot: 15 / 18 later
- Stage-up (clause w / εΔ): same backbone, only if n12 Med improves ≳10% vs that backbone’s step-0

## Ban list

- Free `U_phase^q` / continuous-q diagonal heads
- Per-cavity **private** SNAP+D mixers
- n-C diagonal / nonlocal memorization heads
- New cost operator per layer

## Stop evolve

```bash
touch explore_OCT29/runs/STOP_EVOLVE
```

# TODO_REWRITE — step 0 status

Zach said **go** (2026-10-29). Step 0 is **implemented**.

Target ansatz:

```text
|0⟩ → D(α₀) → [SNAP → D → U_Φ]^L
U_Φ = exp(-i γ_ℓ Φ),  Φ = H_cost   (step 0; later + εΔ)
SNAP+D params shared across cavities; only γ_ℓ varies by layer
```

Param target: **11** per layer (10 shared mixer + 1 γ), independent of C.

---

## Step 0 — DONE

### `lib/ansatz.py`
- [x] Skeleton SNAP→D→`U_Φ` (no free `U_phase^q` in forward)
- [x] Shared `TH` `(L, NU)`, shared `A` `(L, 2)`, `γ` `(L,)`
- [x] Same SNAP+D every cavity; Φ = exact `H_cost`
- [x] `q_from_logit` deprecated helpers only
- [x] Shared α₀ `(2,)` (accepts tiled `(C,2)`); size-norm in GNN

### `lib/gnn.py`
- [x] Dropped `head_q`
- [x] Shared mixer heads; `head_g` → `(L,)`
- [x] `n_params_estimate` / `mixer_params_per_layer()==11` independent of C
- [x] No clause-weight / εΔ heads yet

### `lib/train_loop.py`
- [x] No `q_logits` / `freeze_q` in live path
- [x] Shared SNAP+D + `γ_ℓ` into forward
- [x] `save_gnn_best` meta: `freedom_step=0`, `shared_mixer=True`
- [x] Zero-shot n12 helpers: `transfer_score`, `pick_transfer_state`, `stage_up_ok`

### Scripts / tests / docs
- [x] train / eval_scale / evolve_gnn updated (promote on n12)
- [x] tests green for shared shapes + transfer load
- [x] PLAN / README / RUN flipped to step 0 implemented

---

## Later (not step 0)

1. **Clause weights** — stage-up only if n12 Med improves ≳10% vs same-backbone step-0 (`stage_up_ok`).
2–3. **Local εΔ** — same gate.
4. Protect-small / curriculum if 6→12 still soft.
**Never:** free dense phase, per-cavity private mixers, size-dependent schedules.

# TODO_REWRITE — step 0 first (do not implement until Zach says **go**)

Goal after go: replace the pre-lock free-`U_phase^q` / per-cavity-mixer snapshot with the **locked** skeleton, starting at freedom-ladder **step 0**.

Target ansatz:

```text
|0⟩ → D(α₀) → [SNAP → D → U_Φ]^L
U_Φ = exp(-i γ_ℓ Φ),  Φ = H_cost   (step 0; later + εΔ)
SNAP+D params shared across cavities; only γ_ℓ varies by layer
```

Param target after step 0: **11** per layer (10 shared mixer + 1 γ), independent of C.

---

## Step 0 — pure `H_cost` + shared `(γ_ℓ, SNAP+D_ℓ)`

### `lib/ansatz.py`

- [ ] Change docstring / skeleton to SNAP→D→`U_Φ` (drop free `U_phase^q` path from the forward).
- [ ] Params per layer: shared `TH` shape `(L, NU)` (or `(NU,)` broadcast), shared `A` `(L, 2)`, scalar `γ_ℓ` — **not** `(C, L, …)`.
- [ ] Apply the **same** SNAP+D to every cavity (or one shared cavity channel tiled).
- [ ] Build diagonal phase from **exact `H_cost`** on the product Gray grid (existing A0 cost construction), times `γ_ℓ` only — no `q` power / continuous-q.
- [ ] Remove `q_from_logit` / `q_logits` from the live forward (may leave helpers behind `# deprecated` briefly if tests need a stub).
- [ ] Keep `D(α₀)` init; α₀ may stay per-cavity or shared — prefer **shared or size-normalized** consistent with transfer (document choice in PLAN when implementing).

### `lib/gnn.py`

- [ ] Drop `head_q` (free q).
- [ ] Change mixer heads from per-cavity `C * L * …` to **shared** `L * …` (or global pool → shared vectors).
- [ ] Keep / reshape `head_g` so it emits **one γ per layer** (`L`,) not a single global scalar if the old head was 1-dim for whole circuit — match QAOA-style `γ_ℓ`.
- [ ] Update `n_params_estimate` to reflect **11**-ish layer cost, independent of C.
- [ ] Do **not** add clause-weight or εΔ heads yet (those are steps 1–3).

### `lib/train_loop.py`

- [ ] Stop wiring `q_logits` / `freeze_q` / `fixed_q_logits` into the loss path (delete or no-op).
- [ ] Pass shared SNAP+D + `γ_ℓ` into `MultiCavityAnsatz.forward`.
- [ ] Confirm packs still load; train n=6&9, eval hooks unchanged at API level.
- [ ] Promotion / `save_gnn_best` meta: record `freedom_step=0`, `shared_mixer=True`.

### Scripts

- [ ] `scripts/train.py` — defaults: `--device cpu` friendly; no chaodao assumptions; optional `--L 16` note for FactorGNN path.
- [ ] `scripts/eval_scale.py` — same; document n=12+ packs required.
- [ ] `scripts/evolve_gnn.py` — evolve **arch** only within shared-mixer heads; ban mutations that reintroduce `head_q` or per-cavity mixers.
- [ ] `scripts/gen_teb_packs.py` / `run_walksat_bars.py` — no ansatz change expected.

### Tests

- [ ] `tests/test_ansatz.py` — shapes: shared mixer, no `q_logits`; forward on n=6 smoke.
- [ ] `tests/test_gnn.py` — head output shapes independent of C (or only via graph, not tiled mixers).
- [ ] `tests/test_train_step.py` — one Adam step on CPU without q heads.
- [ ] Keep packing/gray tests as-is.

### Docs after rewrite lands

- [ ] Flip README/PLAN “pre-lock snapshot” banners to “step 0 implemented”.
- [ ] Record new Med/B90 vs so-far bars.

---

## Later (not step 0)

1. **Clause weights** — GNN scalar/vector per clause; Φ = weighted `H_cost` on same graph.  
2–3. **Local εΔ** — residual with cavity-graph sparsity; ε capped.  
4. Protect-small / curriculum if 6→12 transfer still soft.  
**Never:** free dense phase, per-cavity private mixers, size-dependent angle schedules that break fixed-angle concentration.

---

## Gate

Zach: say **`go`** in chat → implement step 0 on box/CPU → pytest green → short train smoke → then step 1+.

# explore_OCT29 — theory lock + campaign plan (2026-10-29)

Snapshot of `qtimes_cost` for the **OCT 29 transfer** track.  
**Step 0 implemented** (Zach go 2026-10-29): shared SNAP+D + `γ_ℓ`, Φ=`H_cost`. Later ladder in [`TODO_REWRITE.md`](TODO_REWRITE.md). Run locus: **box CPU** (chaodao deferred).

---

## 1. Locked ansatz skeleton

Fixed **L across n**:

```text
|0⟩ → D(α₀) → [SNAP → D → U_Φ]^L
```

| Piece | Rule |
|-------|------|
| Mixer | **SNAP + D only**, **shared across cavities** (same SNAP+D params every cavity). Not per-cavity private mixers. |
| Cost | One **Φ per instance**, **same Φ every layer**. `U_Φ = exp(-i γ_ℓ Φ)` with `Φ = H_cost + εΔ` (Δ **local** only). Only **γ_ℓ** changes with layer. |
| “β_ℓ” | Shorthand for **shared SNAP+D params at layer ℓ** — not a separate dial. |
| Ban | Free `U_phase^q` / n-C diagonal heads / nonlocal memorization. |

**Why shared mixers:** per-cavity SNAP+D adds freedom but breaks fixed-angle / light-cone / cavity-MP transfer (size- and site-dependent). Put extra freedom in **local Φ** (clause weights / εΔ), not cavity-private mixers.

**εΔ:** GNN outputs a **local** residual Δ (same sparsity as cavity graph / light cone), then `Φ = H_cost + εΔ`. Not a free dense matrix; keep ε small so it cannot memorize nonlocal phase.

### Freedom ladder (implement in this order)

| Step | Freedom | Status |
|------|---------|--------|
| **0** | Pure `H_cost` + shared `(γ_ℓ, SNAP+D_ℓ)` | **DONE — live** |
| **1** | Learned clause weights on that graph | next |
| **2–3** | + small local `εΔ` | **sweet spot** |
| **4–5** | Nonlocal / free phase | **ban** |

### Param target (planned shared)

- SNAP+D shared → **8 + 2 = 10** mixer + **1 γ_ℓ** per layer → **11**, independent of C.
- **Live now:** shared **11**/layer (independent of C). Old 10C+1 + free `q` removed from forward.

### Train / eval

- Train: **n = 6 & 9**
- Eval: **12 / 15 / 18** (21 optional later)
- Optional two-model path: FactorGNN **L=16** for n≤9; SizeGeneral **L=24** for n≥12
- Keep cavity ansatz; steal QAOA fixed-angle / light-cone ideas (Boulebnane–Montanaro [2208.06909](https://arxiv.org/abs/2208.06909): large **fixed** p OK; fails when p/angles become size-dependent). No full rewrite of the research goal.

### So-far bars (Med / B90) — document only

| n | Med / B90 | Note |
|---|-----------|------|
| 6 | **11.1 / 38.7** | L16 BEST |
| 9 | **8.8 / 34.3** | L16 BEST |
| 12 | **21.1 / 95.9** | L24 SizeGeneral, **trained-on-n12** (not zero-shot) |
| 15 | fails | zero-shot |

Explores on the old chaodao path are stopped for this track.

---

## 2. Code state of this folder

**Step 0 live** (post-go rewrite):

- `lib/ansatz.py` — `|0⟩→D→[SNAP→D→U_Φ]^L`, Φ=`H_cost`, shared SNAP+D, `γ_ℓ` per layer
- `lib/gnn.py` — shared heads (`alpha0`(2,), `A`(L,2), `TH`(L,NU), `gamma`(L,)); no `head_q`
- `lib/train_loop.py` — promote helpers for zero-shot n12; meta `freedom_step=0`
- Evolve: mutate arch only; **promote on n12 Med/B90**; stage-up gate ≥10% later
- Data: `data_n6/n9` + generate `data_n12` for promote gate

See [`RUN.md`](RUN.md).

---

## 3. Locus & sync

| Role | Path |
|------|------|
| Box / git source of truth | `/workspace/qumode_sep/explore_OCT29` |
| Mac clone | `~/qumode_sep/explore_OCT29` |
| Cloud | same package; `--device cuda:N` when available |
| Chaodao | **deferred** |

Author/docs on box or Mac; `git pull` on Mac after push. Prefer **direct push to `origin/main`** for this folder.

---

## 4. Layout

```text
explore_OCT29/
  PLAN.md TODO_REWRITE.md RUN.md README.md
  requirements.txt  __init__.py
  lib/       # ansatz, gnn, train_loop, sat_gen, packing, gray_grid, walksat, device
  scripts/   # gen_teb_packs, run_walksat_bars, train, eval_scale, evolve_gnn
  tests/
  data/      # data_n{n}.npz
  baselines/
  runs/      # gnn_best/, STOP_EVOLVE, evolve CSV
```

Self-contained: **no imports outside `explore_OCT29/`**.

---

## 5. Code order when Zach says go

1. Pure `H_cost` + shared `(γ_ℓ, SNAP+D_ℓ)` baseline  
2. Clause weights  
3. Tiny local `εΔ`  
4. Protect-small / curriculum if transfer still soft  

Details: [`TODO_REWRITE.md`](TODO_REWRITE.md).


## Promote gate (evolve)

- Primary: **zero-shot n12** `S = (log Med + log B90)/2` (lower better).
- Train-n Med/B90 logged but **never** alone enough to overwrite `runs/gnn_best/`.
- Stage-up (clause weights / εΔ): same backbone, only if n12 Med improves ≳ **10%** vs that backbone’s step-0 (`stage_up_ok`).
- Stop: `touch explore_OCT29/runs/STOP_EVOLVE`.

---

## 6. Long-confirm + pivot (2026-10-29, Asia/Shanghai)

### Long-confirm of micro champ (H=64,r6,mean,residual+degree, lr=1e-3)

Train: 1500 steps, L=5, seed=0, max_val/n12=24. Best step **1220**. Snapshot: `runs/gnn_best_long_confirm/` (micro frozen at `runs/gnn_best_micro_t223/`).

| split | n6 Med/B90 | n9 Med/B90 | n12 Med/B90 (zero-shot) |
|-------|------------|------------|-------------------------|
| train-best val B=24 | **11.40 / 31.37** | **59.20 / 281.2** | **4.05e8 / 2.62e11** |
| eval val B=32 | 11.40 / 27.88 | 89.01 / 750.7 | 4.05e8 / ~1e12 |
| eval test B=32 | 11.31 / 32.18 | 157.5 / 898.8 | 1.11e8 / 9.19e11 |

Growth (test Med9/Med6 ≈ **13.9**, ~≥2³). n12 transfer **collapsed**.

### vs OCT20 fair global-cost L4 (≈1.26ⁿ class)

| | n8 | n12 | n16 |
|--|----|-----|-----|
| OCT20 Med/B90 | 4.76/8.81 | 12.31/26.31 | 31.19/74.14 |

Recipe gap (honest): OCT20 = calibrated Gray-digit X mixer (SNAP–D–SNAP ≈ ∏exp(−iβX/2)) + warm γ/β + optional per-layer clause weights. OCT29 long-confirm = **cold** shared SNAP+D + shared H_cost only, QAOA-like sandwich, GNN-emitted heads.


**Zach (2026-10-29 ~17:32 CST):** further long-confirm of this sandwich declared useless — do **not** resume 1500-step GNN confirm. Pivot immediately to native gate-sequence explore (`evolve_seq`). Numbers above kept only as negative baseline.

### Verdict on GNN-hyper evolve

**Pause.** Micro already plateaued (~200 trials, one H=64 r6 basin). Long confirm shows train-n can look “ok” while **zero-shot n12 is useless**. Lever is **not** more 15-step MP-GNN arch search.

### Main direction (Zach 2026-10-29): native gate-sequence explore

Theory lock holds only for the completed confirm baseline. Going forward, prioritize **qumode-native, hardware-efficient gate sequences** that could train-small→transfer-large — **not** recreating QAOA with SNAP+D labels, and **not** more GNN micro-evolve.

Still ban: free `U_phase^q` / dense nonlocal phase heads with no native payoff.

### Next probes (approved to run nonstop)

1. **Gate-sequence evolve** (`scripts/evolve_seq.py`, `lib/seq_ansatz.py`) — mutate prep/mixer/COST ordering; SNAP–D–SNAP sandwiches; shared fixed-angle params (no GNN). Promote only on clear n12 Med/B90 improvement vs long-confirm baseline. ← **running now**
2. **Gray-X-like calibrated / structured mixer** motif as first-class native block (shared strengths), not QAOA emulation.
3. **Prep motifs** (SNAP+D flat-ish prep) + cost placement variants.
4. Freedom-1 clause weights / εΔ **only if** a native sequence already moves n12 off the 1e8 cliff.
5. Keep micro GNN evolve **stopped** until a sequence baseline transfers.

Stop seq loop: `touch explore_OCT29/runs/STOP_SEQ_EVOLVE`.


---

## 7. Opus probe 1 — driver bake-off (2026-10-29 ~17:41 CST)

Two arms only (Ideal Gray-X dropped per Zach). Results: `runs/driver_bakeoff/`.

| arm | n9 Med/B90 | n12 | n15 | b_Med | b_B90 |
|-----|------------|-----|-----|-------|-------|
| **projector L4 trained** | **9.89/25** | **36.8/65.3** | **76.5/152** | **1.406** | **1.352** |
| snap_d COST→SNAP→D L4 trained | 289/6.24e3 | 8.89e3/1.57e5 | 1.24e5/1.11e6 | 2.746 | 2.371 |

Winner: native projector `W·SNAP₀(β)·W†` (W flat-prep F=0.999). Keeps **b≪2**. Next: probe 2 (train {9,12}).


## 8. Opus probe 2 — train {n9,n12} (2026-10-29 ~17:44 CST)

| model | n9 | n12 | n15 | n18 | b_Med | Med15/12 | Med18/15 |
|-------|----|-----|-----|-----|-------|----------|----------|
| n9-only bake-off | 9.27/28.3 | 36.8/65.3 | 76.5/152 | 373/996 | 1.483 | 2.08 | 4.88 |
| **train {9,12}** | **6.03/19.9** | **17.5/40.4** | **30.9/75.3** | **149/581** | **1.405** | **1.76** | **4.84** |

Transfer healthy (b≪2, no Med15 cliff). Results: `runs/probe2_train912/`. Next: probe 3 Fourier.


## 9. Opus probe 3 — Fourier depth-at-deploy (2026-10-29 ~17:47 CST)

q=2 modes trained at L=4 on {9,12}; deploy L∈{4,6,8,12} no retrain.

| n \ L | 4 | 6 | 8 | 12 |
|-------|---|---|---|-----|
| 15 Med/B90 | 50.2/102 | 25.1/37 | 16/21.5 | **9.67/14.6** |
| 18 Med/B90 | 194/674 | 88.6/361 | 77.5/688 | 75.4/481 |

n15 Med falls monotonically with L. b_Med≈1.41–1.45. Results: `runs/probe3_fourier/`.


## 10. Opus probe 5 — W-atom seq vocab (started 2026-10-29 ~17:50 CST)

Vocab: `W`, `WDAG`, `FLAT_PREP`, `SNAP0`, `PROJ` (=W·SNAP0·W†). Seeded from projector.
Promote on fitted **b** over {9,12,15}. Loop: `scripts/probe5_seq_vocab.py`, stop via `runs/STOP_PROBE5`.
Results: `runs/probe5_seq_vocab/`. Old raw `evolve_seq` stopped.

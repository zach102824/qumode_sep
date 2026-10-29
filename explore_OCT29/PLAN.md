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

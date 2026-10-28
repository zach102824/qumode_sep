# qtimes_cost — self-contained campaign plan

- Updated: 2026-10-28 ~16:55 Asia/Shanghai (Zach: **self-contained folder**; ladder **n∈{6,9,12,15,18,21}**; train **6&9**; eval **12–21**; M0→M1→M2; Day2 cancelled)
- Status: **plan only** — no GPU jobs. Next work = code under this folder only.
- Day2 / tied-cost farm: **cancelled**. This track owns the next chaodao runs.

---

## Folder layout (locked)

| Role | Path |
|---|---|
| Box / git source of truth | `/workspace/qumode_sep/qtimes_cost` |
| Mac clone (author code) | `~/qumode_sep/qtimes_cost` |
| Chaodao GPU run folder | Prefer `F:\zach\qtimes_cost` (WSL `/mnt/f/zach/qtimes_cost`); GPU alias `F:\zach\GPU\qtimes_cost` OK if that is where qumode_GPU parks |
| Data packs | `qtimes_cost/data/data_n{n}.npz` |
| Classical bars | `qtimes_cost/baselines/walksat_n{n}.json`, `wslm_n{n}.json` |

Sibling of `oct_explore/` — **not** under `oct_explore/`.

### Self-contained rule (hard)

- **All runnable code lives only under `qtimes_cost/`.**
- May **copy / vendor** prior WalkSATlm, TEB generators, sim helpers, GNN pieces into this folder.
- Scripts **must not** `import` or `sys.path` into `oct_explore/` (or any other sibling). No live calls outside the folder.
- Vendored copies are OK and expected for cleanliness.

Suggested package layout (to create):

```
qtimes_cost/
  PLAN.md                 # this file
  README.md
  __init__.py
  lib/                    # vendored sim, Gray grid, WalkSAT wrappers, GNN, ansatz
  scripts/
    gen_teb_packs.py      # M0
    run_walksat_bars.py   # M1
    train.py              # M2
    eval_scale.py         # M2
  data/                   # data_n{n}.npz
  baselines/              # walksat / wslm JSON
  runs/                   # checkpoints / logs
```

---

## Zach-facing plan (edited)

1. **Brand-new self-contained folder** `qtimes_cost/` at repo root. Copy what you need from prior tracks; **no external imports**.
2. **Freeze GNN** from prior data: bipartite message-passing, **H=128, rounds=4**, mean aggregation (+ degree). No arch search this campaign.
3. **Ansatz:** `|0⟩ → D(α₀) → [SNAP_ℓ → D(α_ℓ) → U_phase^{q_ℓ}]^L` on **C = n/3 cavities** (fixed K=3 bits/cav).
   - One shared `U_phase` per formula, applied `q` times per layer.
   - `q_ℓ ∈ {1…10}` is a **global schedule** (shared across formulas), continuous→anneal→integer. Report integer-q only.
   - **A0:** exact cost Hamiltonian phase. **A1:** cost prior + GNN clause weights + per-clause 8-phase residuals (zero-init ⇒ A1 starts = A0). Promote A1 only if earned.
4. **n ladder:** generate Hamiltonians / TEB for **n∈{6,9,12,15,18,21}**. **Train 6&9**; **eval 12,15,18,21**. Chaodao is fast → include **n=21**.
5. **Pipeline order (locked):**
   - **M0** — generate TEB / Hamiltonian packs for all n on the ladder
   - **M1** — classical WalkSAT / WalkSATlm scaling bars on the **same seeds**
   - **M2** — ansatz code + train on 6&9 + scale-eval on 12/15/18/21
6. Opus extras still apply:
   - **Regression anchor:** new code with `q≡1`, `α₀=0`, and a tiny parity harness should roughly reproduce Night1 spirit before trusting multi-cavity.
   - **q-oracle:** best integer q per instance with everything else fixed — decides whether GNN-predicted q (A2) is worth building.

**Opus calls (kept):**

| Question | Call |
|---|---|
| Exact cost vs fine-tune? | **Exact first (A0)**; promote A1 only if train-n B90 ≥15% better **and** n12 not hurt |
| Shared q vs GNN-q? | **Global shared schedule first**; A2 only if oracle ≥25% B90 gain |
| Multi-n mix? | Weights **n6 0.35 / n9 0.65**; curriculum: n6 first ~20%, then ramp n9 |
| Mixer per n? | **Shared size-normalized mixer** is the headline; per-n mixer = control only |
| Coherent start? | Keep `D(α₀)`; credit only with `α₀=0` ablation |

---

## Cavity packing (fixed K=3 bits per cavity)

**Hard lock:** every cavity has **K = 3** Gray Fock digits and **NU = 8** computational levels. Number of cavities scales with problem size:

\[
C = n / 3 \quad (n \bmod 3 = 0),\qquad \text{state dim} = 8^{C} = 8^{n/3} = 2^{n}.
\]

Bit assignment (product Gray grid):

- Cavity `c = 0 … C-1` owns variables `x_{3c}, x_{3c+1}, x_{3c+2}` (Gray MSB…LSB inside that cavity’s Fock level).
- Full bitstring = concatenation of Gray-decoded Fock levels across cavities.

| n | C | K/cav | NU | state dim `8^C` | role |
|--:|--:|------:|---:|----------------:|------|
| 6 | 2 | 3 | 8 | 64 | **train** |
| 9 | 3 | 3 | 8 | 512 | **train** |
| 12 | 4 | 3 | 8 | 4096 | **eval** |
| 15 | 5 | 3 | 8 | 32768 | **eval** (heavy / chunked OK) |
| 18 | 6 | 3 | 8 | 262144 | **eval** (chunked / approx) |
| 21 | 7 | 3 | 8 | **2 097 152** | **eval** (chunked / approx — chaodao-fast; no dense full-state train) |

Suggest `NF ≥ NU` per cavity (e.g. 24–40) with a larger `NF_CHK` for leakage checks.

**Layer ops (multi-cavity):**

- Per layer, per cavity: independent **SNAP(θ_{ℓ,c})** then **D(α_{ℓ,c})**.
- Init: free coherent **D(α₀)** — start **shared / size-normalized**; ablate per-cavity.
- **`U_phase^q`:** one diagonal phase on the **product** Gray grid (full `8^C` computational basis). Only joint op in this track.
- **No ECD / transmon / inter-cavity beamsplitter** unless Zach adds later.
- A0 builds `U_phase` from the SAT cost on the product grid; A1 adds bounded clause residuals — **not** a raw free `2^n` diagonal head.

---

## Pipeline milestones

### M0 — generate Hamiltonians / TEB packs (FIRST)

**Why:** prior packs only cover n=6 (and n=8 elsewhere). **n=9,12,15,18,21 never existed** — must be generated inside this folder before classical bars or ansatz training.

| Need | Role |
|---|---|
| `data/data_n6.npz` | train (copy/regenerate with **vendored** generator for bit-identical or documented seed parity) |
| `data/data_n9.npz` | train |
| `data/data_n12.npz` | eval |
| `data/data_n15.npz` | eval |
| `data/data_n18.npz` | eval (clauses+seeds+nsol; dense `E` optional / chunked) |
| `data/data_n21.npz` | eval (same; **no** dense full-state train) |

**Generator rules (vendored into `qtimes_cost/lib/`):**

- Same spirit as historical TEB: `m = round(9.93 * n)`, 4-SAT, unique-GS filter `E.min()==0` where `E` is stored.
- Seed bases: keep documented TRB/TEB/VAB (or re-document if regenerating n6 from scratch).
- Counts default: train 10k / val 200 / test 100 for n≤12; for n≥15 allow smaller splits or clauses+seeds+nsol only if dense `E` is too big. **Train n6/n9 need full `E`.**

**Script:** `qtimes_cost/scripts/gen_teb_packs.py` — **vendored** gen only; no import from `oct_explore`.

**Gate:** packs for **n=6 and n=9** exist under `qtimes_cost/data/` before M1/M2 GPU work; at least n=12 pack ready for scaling eval. n=15/18/21 packs can land in parallel with early M2 but must exist before those evals.

### M1 — classical WalkSAT / WalkSATlm scaling bars

- Run **vendored** WalkSAT + WalkSATlm on the **same test seeds** as M0 packs.
- Write `baselines/walksat_n{n}.json` and `wslm_n{n}.json` for **n∈{6,9,12,15,18,21}**.
- Gate: n=6 Med/B90 reproduces prior campaign bar within noise; n=9 bar exists; n=12+ bars exist before claiming scale wins.
- Headline classical curve = Med/B90 vs n (log-y) for WalkSAT and WalkSATlm — the bar the ansatz must track or beat.

**Script:** `qtimes_cost/scripts/run_walksat_bars.py` (vendored binary/wrapper + JSON writer).

### M2 — ansatz code, train, scale-eval

1. Implement circuit: `|0⟩ → D(α₀) → [SNAP → D → U_phase^q]^L` with packing above.
2. GNN H=128 / r=4 frozen arch; heads for clause weights / residuals / γ scale as needed.
3. **Train** on n=6&9 only (weights ~0.35/0.65).
4. **Eval** on n=12,15,18,21 (chunked / approximate forward at large C).
5. Report Med, B90, R(n)=Med_Q/Med_WS vs classical bars from M1 at **fixed L**.

**Scripts:** `train.py`, `eval_scale.py` under `qtimes_cost/scripts/` (or `python -m qtimes_cost.train`).

---

## Spec locks (short)

- **Circuit:** coherent start + L layers of (per-cavity SNAP → D) → product `U_phase^q`.
- **U_phase:** never a free raw 2ⁿ diagonal head. A0 = cost phase; A1 = cost prior + bounded residuals.
- **q:** `q̃ = 1 + 9·sigmoid(u)`; anneal free → sin² penalty → hard round + polish. Report integer q only.
- **Mixer:** α scaled by √N (N=2ⁿ); SNAP as smooth fn of r=√(k/NU) so params transfer as **C grows** at fixed NU=8.
- **GNN:** H=128, r=4, mean+degree.
- **Loss:** `log(1/p) − log(Med_WS(n))` + CVaR₉₀; τ ∈ {1.0, 1.6}. Leakage penalty on guard band.
- **L ∈ {5, 8}** fixed across n for the headline. Don’t go past L=8 before the q-oracle.
- **Hardware encoding:** **K=3, NU=8, C=n/3**. Odd n / non-multiples of 3 out of scope.

---

## Mac → chaodao workflow (future default)

Author on **Mac**; sync whole `qtimes_cost/` (code + data + baselines); GPUs only on chaodao **after M0 (and ideally M1)**.

```bash
# on Mac, from repo root (host alias `chaodao` in ~/.ssh/config)
rsync -avz --delete qtimes_cost/ chaodao:/mnt/f/zach/qtimes_cost/
# if using the GPU subfolder instead:
# rsync -avz --delete qtimes_cost/ chaodao:/mnt/f/zach/GPU/qtimes_cost/
```

```powershell
# on chaodao (PowerShell) — match the sync target
cd F:\zach\qtimes_cost
# python -m qtimes_cost.scripts.gen_teb_packs --n 6,9,12,15,18,21   # M0
# python -m qtimes_cost.scripts.run_walksat_bars --n 6,9,12,15,18,21 # M1
# python -m qtimes_cost.train ...                                    # M2
```

```bash
# on chaodao (WSL)
cd /mnt/f/zach/qtimes_cost
# python -m qtimes_cost.scripts.gen_teb_packs --n 6,9,12,15,18,21
```

Align path with wherever qumode_GPU parks (`F:\zach\qtimes_cost` vs `F:\zach\GPU\qtimes_cost`) before first sync.

---

## Dual-GPU / farm order (when Zach says go; **no jobs yet**)

| Phase | Work |
|---|---|
| **0** | **M0** gen packs n∈{6,9,12,15,18,21} (CPU; chaodao OK for speed) |
| **1** | **M1** WalkSAT/wslm bars on same seeds → classical scaling curve |
| **2** | M2 simulator parity + A0 sweep on **n6+n9**: L∈{5,8} × τ∈{1.0,1.6} × seeds; ablations (q≡1, α₀=0) |
| **3** | A1 main if gate; **scale-eval n=12,15,18,21** (chunked at ≥15) |
| **4** | A2 / GNN-q only if oracle gate passes |

**Kill / promote (val; test once per promoted cfg):**

- Early kill if n9 Med ≫ WalkSAT×2 (exact once M1 bars exist)
- Kill if Med growth vs n much worse than WalkSAT growth
- Promote: integer-q beats / tracks WalkSAT Med on n9 and keeps Med_n / Med_9 near WalkSAT’s growth
- Headline: T2/T3 in ≥2/3 seeds — report **median across seeds**

**Success tiers (recalibrate after M1 bars):**

- **T1** better amortization than old tied-cost lock on train n
- **T2** classical parity @n9: Med/B90 ≤ WalkSAT bars
- **T3 (goal):** eval n Med/B90 ≤ WalkSAT; **Med_{n}/Med_9 ≤ WalkSAT growth**; R(n)=Med_Q/Med_WS flat or falling across **12,15,18,21** at **fixed L**

---

## What NOT

- **Day2 cancelled** — do not wait on it or schedule around it.
- Don’t put this track under `oct_explore/`.
- **Don’t import / call code outside `qtimes_cost/`** — vendor copies only.
- Don’t use K=n/2 or fixed C=2 packing for the headline.
- Don’t train primarily on n=12/15/18/21.
- Don’t start A0 / long GPU jobs before M0 packs for train n (6,9) and at least M1 bar for n=12.
- No raw 2ⁿ diagonal, Gumbel-q, n6-only screening as the only result, continuous-q as reported numbers, per-n mixer as headline, H=256 / arch search, sum aggregation, L>8 before oracle, dense full-state **train** at n≥15 (esp. n=21), mean-1/p loss, mismatched WS generators at new n.
- No ECD/transmon unless Zach adds later.
- Don’t run GPUs from Mac — sync to chaodao first.
- Don’t message qumode_GPU / start farm until Zach says go **and** M0 is green for train n.

Historical Opus notes under `oct_explore` are superseded by this file for packing / train-n / folder rules.

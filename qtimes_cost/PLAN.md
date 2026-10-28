# qtimes_cost — new ansatz + fixed-3-bits multi-cavity scaling

- Updated: 2026-10-28 ~16:45 Asia/Shanghai (Zach: folder at repo root; **K=3 bits/cavity fixed**; train 6&9; eval 12/15/18+)
- Prior Opus spirit kept: A0 → A1, global `q∈{1…10}`, GNN H=128/r=4 freeze, coherent `D(α₀)`, Mac→chaodao sync, **do not stop Day2**
- Status: **plan only** — no GPUs, no qumode_GPU ping, no full code yet

---

## Folder layout (locked)

| Role | Path |
|---|---|
| Box / git source of truth | `/workspace/qumode_sep/qtimes_cost` |
| Mac clone (author code) | `~/qumode_sep/qtimes_cost` |
| Chaodao GPU run folder | `F:\zach\GPU\qtimes_cost` (WSL: `/mnt/f/zach/GPU/qtimes_cost`) |

Sibling of `oct_explore/` — **not** under `oct_explore/`. Day2 / tied-cost trees stay untouched.

---

## Edited plan (Zach-facing)

1. **Brand-new folder** `qtimes_cost/` at repo root. Copy/freeze only formula encoding + GNN pieces you need; never import Day2; never stop Day2.
2. **Freeze the GNN** from prior data: bipartite message-passing, **H=128, rounds=4**, mean aggregation (+ degree features). No architecture search this campaign.
3. **Ansatz:** `|0⟩ → D(α₀) → [SNAP_ℓ → D(α_ℓ) → U_phase^{q_ℓ}]^L` on **C = n/3 cavities** (see packing).
   - One shared `U_phase` per formula (same at every layer), applied `q` times.
   - `q_ℓ ∈ {1…10}` is a **global schedule** (all formulas share it), trained continuous then annealed to integers. Report integer-q only.
   - **A0:** exact cost Hamiltonian phase. **A1:** cost prior + GNN clause weights + per-clause 8-phase residuals (zero-init so A1 starts = A0). Promote A1 only if earned (see gates).
4. **Train on n = 6 and n = 9 only.** Eval / scaling on **n = 12, 15, 18, …** (and optionally re-eval train sizes). Do **not** train on n=12 as primary. Headline = Med/B90 growth vs WalkSAT at fixed L with a **size-normalized shared mixer**.
5. **Data milestone first:** we do **not** yet have TEB packs or WalkSAT bars for n∈{9,12,15,18}. Generate those (same generator / clause ratio / GS filter as `data_n6.npz`) before any train. See § Data inventory.
6. Two Opus extras still apply:
   - **Regression anchor:** new code with `q≡1`, `α₀=0`, and the old 2-cavity K=n/2 packing on n=8 (or a tiny parity harness) should roughly reproduce Night1 (~12.4/84.6) before trusting the new multi-cavity path.
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
- Full bitstring = concatenation of Gray-decoded Fock levels across cavities (same spirit as `tiecost_lib.GRID_TO_X`, but **C varies**, K fixed).

| n | C | K/cav | NU | state dim `8^C` | role |
|--:|--:|------:|---:|----------------:|------|
| 6 | 2 | 3 | 8 | 64 | **train** |
| 9 | 3 | 3 | 8 | 512 | **train** |
| 12 | 4 | 3 | 8 | 4096 | **eval** |
| 15 | 5 | 3 | 8 | 32768 | **eval** (heavy) |
| 18 | 6 | 3 | 8 | 262144 | **eval** (chunked / approx OK) |

Suggest `NF ≥ NU` per cavity (e.g. 24–40) with a larger `NF_CHK` for leakage checks, same as prior tracks.

**Layer ops (multi-cavity):**

- Per layer, per cavity: independent **SNAP(θ_{ℓ,c})** then **D(α_{ℓ,c})** (local mixer on that mode).
- Init: free coherent displace **D(α₀)** — either one shared α₀ across cavities or per-cavity α₀,c (start with **shared / size-normalized**; ablate).
- **`U_phase^q`:** one diagonal phase on the **product** Gray grid (full `8^C` computational basis). This is the only place cavities talk to each other in this track.
- **No ECD / transmon / inter-cavity beamsplitter** unless Zach adds them later. Joint structure = shared cost-phase only.
- Opus-style **cost prior** still applies: A0 builds `U_phase` from the SAT cost on the product grid; A1 adds bounded clause residuals predicted by the GNN — not a raw free `2^n` diagonal head.

---

## Data inventory & generation milestone (blocker before train)

### What already exists

| Artifact | Path | Notes |
|---|---|---|
| TEB pack n=6 | `oct_explore/results/oct23_exp/data_n6.npz` | n=6, m=60 (ratio 10), train 10k / val 200 / test 100; keys `*_E`, `*_nsol`, `*_cls_v/s`, seeds |
| TEB pack n=8 | `oct_explore/results/oct23_exp/data_n8.npz` | n=8, m=79; **not** used as primary train in this track (parity / anchor only) |
| WalkSAT JSON n=6,8,10 | `oct_explore/results/classical_4sat_head/raw/ubc_walksat_n{6,8,10}.json` | campaign Med/B90 bars historically quoted separately (n6 ~8.5/16, n8 ~14/29, n12 ~32.5/76 from prior notes) |

### What does **not** exist (must create)

| Need | Status |
|---|---|
| `data_n9.npz` | **missing** — create |
| `data_n12.npz` | **missing** under `oct_explore/results/oct23_exp/` (no TEB pack found anywhere in repo) — create |
| `data_n15.npz`, `data_n18.npz` | **missing** — create (eval; E arrays are 2^n so disk/RAM heavy) |
| WalkSAT / WalkSATlm bars for n∈{9,12,15,18} | **missing** — create with **same generator, clause ratio (~9.875–10), GS filter, and lookup protocol** as prior classical_4sat_head |

### Milestone M0 (CPU / farm OK; no train GPUs)

1. Reuse `sim.gen_instance` / `sim.instances` (or the oct23_exp generator that built `data_n6.npz`) with the **same** TEB seed bases and unique-GS filter.
2. Emit `qtimes_cost/data/data_n{n}.npz` for n∈{6,9,12,15,18} (n=6 may symlink/copy the existing pack for reproducibility).
3. For n≥15: store clauses + seeds + nsol always; full dense `E` optional / chunked if RAM hurts — but train path for n6/n9 needs full `E` as today.
4. Run WalkSAT (and WalkSATlm if that was the campaign bar) on the **same test seeds**; write `qtimes_cost/baselines/walksat_n{n}.json` with Med / B90 lookups.
5. Gate: n=6 Med/B90 must reproduce prior campaign bar within noise before trusting n=9+.

**Do not start A0 training until M0 lands for n=6 and n=9 (train) plus at least n=12 WalkSAT bar (eval).**

---

## Spec locks (short)

- **Circuit:** coherent start + L layers of (per-cavity SNAP → D) → product `U_phase^q`.
- **U_phase:** never a free raw 2ⁿ diagonal head. A0 = `exp(−iγ Σ unsat)`; A1 = cost prior + bounded residuals.
- **q:** `q̃ = 1 + 9·sigmoid(u)`; anneal free → sin² penalty → hard round + polish. Report integer q only.
- **Mixer:** α scaled by √N (N=2ⁿ); SNAP as smooth fn of r=√(k/NU) (Chebyshev) so params transfer as **C grows** at fixed NU=8.
- **GNN:** H=128, r=4, mean+degree. Heads: clause w_c + 8-phase residual; graph readout for γ scale.
- **Loss:** `log(1/p) − log(Med_WS(n))` + CVaR₉₀; τ ∈ {1.0, 1.6}. Leakage penalty on guard band.
- **L ∈ {5, 8}** fixed across n for the headline. Don’t go past L=8 before the q-oracle.
- **Hardware encoding:** **K=3, NU=8, C=n/3** as in § Cavity packing. Odd n / non-multiples of 3 out of scope.

---

## Mac → chaodao workflow (future default)

Author on **Mac**; sync to chaodao `qtimes_cost`; GPUs only run there.

```bash
# on Mac, from repo root after edits (host alias `chaodao` in ~/.ssh/config)
rsync -avz --delete qtimes_cost/ chaodao:/mnt/f/zach/GPU/qtimes_cost/
# or:
# scp -r qtimes_cost chaodao:F:/zach/GPU/
```

```powershell
# on chaodao (PowerShell)
cd F:\zach\GPU\qtimes_cost
# python -m qtimes_cost.train ...   # TBD when package lands
```

```bash
# on chaodao (WSL)
cd /mnt/f/zach/GPU/qtimes_cost
# python -m qtimes_cost.train ...
```

Train CLI remains **TBD** until the package exists under this folder.

---

## Dual-GPU run order (when free; **never preempt Day2**)

| Phase | GPU-A | GPU-B |
|---|---|---|
| 0 | M1 simulator / JAX parity on n=6 (C=2) | M0 data gen + WalkSAT lookups (CPU OK) for n=9,12,… |
| 1 | M2 regression / packing sanity | M4 **A0** sweep on **n6+n9**: L∈{5,8} × τ∈{1.0,1.6} × 2 seeds |
| 2 | M5 **A1** main (best A0 L, 3 seeds) | Ablations (q≡1, α₀=0, per-n mixer) + M6 q-oracle |
| 3 | A2 only if oracle gate passes; else more A1 seeds | M7 scaling eval **n=12,15,18** (chunked at 15/18) |

**Kill / promote (val only; test once per promoted cfg):**

- @25% steps: kill if n9 Med ≫ WalkSAT×2 (set exact once M0 bars exist)
- @50%: kill if Med12/Med9 > ~3× WalkSAT growth ratio
- Promote: integer-q beats WalkSAT Med on n9 and keeps Med12/Med9 near WalkSAT’s growth
- Headline: T2/T3 in ≥2/3 seeds — report **median across seeds**

**Success tiers (recalibrate numbers after M0 WalkSAT bars):**

- **T1** beat old lock spirit: clearly better amortization than Night1-style tied cost on train n
- **T2** classical parity @n9: Med/B90 ≤ WalkSAT bars
- **T3 (goal):** n12 Med/B90 ≤ WalkSAT; **Med12/Med9 ≤ WalkSAT’s Med12/Med9**; R(n)=Med_Q/Med_WS flat or falling across n=12,15,18 at **fixed L**

---

## What NOT

- Don’t stop / edit Day2.
- Don’t put this track under `oct_explore/`.
- Don’t use K=n/2 or fixed C=2 packing for the headline (that was the old plan).
- Don’t train primarily on n=12/15/18.
- No raw 2ⁿ diagonal, Gumbel-q, n6-only screening as the only result, continuous-q as reported numbers, per-n mixer as headline, H=256 / arch search, sum aggregation, L>8 before oracle, dense full-state train at n≥15, mean-1/p loss, mismatched WS generators at new n.
- No ECD/transmon in this track unless Zach adds later.
- Don’t run GPUs from Mac — sync to chaodao `qtimes_cost` first.
- Don’t message qumode_GPU / start farm until Zach says go.

Full Opus detail (older 2-cavity draft): `oct_explore/v3/openevolve/runs/n8_phase1/opus_newfolder_scale_ideas.md` — packing/train-n there are **superseded** by this file.

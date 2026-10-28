# New ansatz + multi-n scaling plan

- Source: Opus 5.5 high (`opus_newfolder_scale_ideas.md`), 2026-10-28 ~16:04 Asia/Shanghai
- Track: **brand-new folder** `oct_explore/new_ansatz_scale/` (do **not** continue Day2 / tied-cost tree; do **not** stop Day2)
- Status: plan only — **no GPUs started**, qumode_GPU not messaged

---

## Edited plan (Zach-facing)

1. **New folder** `oct_explore/new_ansatz_scale/`. Copy/freeze only what is needed (formula encoding, GNN). Never import from the Day2 tree; never touch Day2.
2. **Freeze the GNN** from prior data: bipartite message-passing, **H=128, rounds=4**, mean aggregation (+ degree features). No architecture search this campaign.
3. **Ansatz:** `|0⟩ → D(α₀) → [SNAP_ℓ → D(α_ℓ) → U_phase^{q_ℓ}]^L`
   - One shared `U_phase` per formula (same at every layer).
   - `q_ℓ ∈ {1…10}` is a **global schedule** (all formulas share it), trained continuous then annealed to integers. Only integer-q numbers are reported.
   - **A0:** exact cost Hamiltonian. **A1:** cost prior + GNN clause weights + per-clause 8-phase residuals (zero-init so A1 starts = A0).
4. **Train on n = 6, 8, 12 together** (n6 low weight). Hold out **n=10** (interpolation) and **n=14, 16** (extrapolation). Headline = Med/B90 growth vs WalkSATlm at fixed L, with a **size-normalized shared mixer** that can transfer to unseen n.
5. Two extras Zach’s draft lacked:
   - **Regression anchor:** new code with `q≡1`, `α₀=0` must roughly reproduce Night1 (~12.4/84.6).
   - **q-oracle:** best integer q per instance with everything else fixed — decides whether GNN-predicted q (A2) is worth building.

**Opus’s key opinions (where the draft was vague):**
| Question | Call |
|---|---|
| Exact cost vs fine-tune? | **Exact first (A0)**; promote A1 only if n8 B90 ≥15% better and n12 not hurt |
| Shared q vs GNN-q? | **Global shared schedule first**; A2 only if oracle ≥25% B90 gain |
| Multi-n mix? | Weights **n6 0.15 / n8 0.40 / n12 0.45**; curriculum: n6+n8 first 20%, then ramp n12 |
| Mixer per n? | **Shared size-normalized mixer** is the headline; per-n mixer = control only |
| Coherent start? | Keep `D(α₀)`, but credit only with `α₀=0` ablation (old SNAP→D on vacuum already started coherent after first D) |

---

## Spec locks (short)

- **Circuit:** coherent start + L layers of SNAP → D → `U_phase^q`.
- **U_phase:** never a free raw 2ⁿ diagonal. A0 = `exp(−iγ Σ unsat)`; A1 = cost prior + bounded residuals.
- **q:** `q̃ = 1 + 9·sigmoid(u)`; anneal free → sin² penalty → hard round + polish. Report integer q only.
- **Mixer:** α scaled by √N (N=2ⁿ); SNAP as smooth function of r=√(k/N) (Chebyshev) so params transfer across n.
- **GNN:** H=128, r=4, mean+degree (not sum). Heads: clause w_c + 8-phase residual; graph readout for γ scale.
- **Loss:** `log(1/p) − log(Med_WS(n))` + CVaR₉₀; τ ∈ {1.0, 1.6}. Leakage penalty 0.14 on guard band (cutoff 1.25N).
- **L ∈ {5, 8}** fixed across n for the headline. Don’t go past L=8 before the q-oracle.

Bars (lookups): n6 8.47/16 | n8 14/29.4 | n12 32.5/76.1. Night1 old lock: 12.36/84.6.

---

## Dual-GPU run order (when free; never preempt Day2)

| Phase | GPU-A | GPU-B |
|---|---|---|
| 0 | M1 simulator / JAX parity | (M0 WalkSAT lookups on CPU) |
| 1 | M2 regression anchor → M3 ceilings | M4 **A0** sweep: L∈{5,8} × τ∈{1.0,1.6} × 2 seeds |
| 2 | M5 **A1** main (best A0 L, 3 seeds) | Ablations (q≡1, α₀=0, per-n mixer) + M6 q-oracle |
| 3 | A2 only if oracle gate passes; else more A1 seeds | M7 scaling eval n=10/14/16 |

**Kill / promote (val only; test once per promoted cfg):**
- @25% steps: kill if n8 Med > 20 or n12 Med > 60
- @50%: kill if n8 B90 > 80 or Med12/Med8 > 3.0
- Promote: integer-q n8 Med ≤ 13, B90 ≤ 50, n12 Med ≤ 40
- Headline: T2/T3 (§ below) in ≥2/3 seeds — report **median across seeds**, track fresh-start F1

**Success tiers:**
- **T1** beat old lock: n8 Med < 12.36 and B90 < 60
- **T2** classical parity @n8: Med ≤ 14 and B90 ≤ 35
- **T3 (goal):** n12 Med ≤ 32.5 / B90 ≤ 76; Med12/Med8 ≤ 2.3; R(n)=Med_Q/Med_WS flat or falling across n=10,14,16 at **fixed L**

---

## What NOT (reminders)

Don’t stop/edit Day2. No raw 2ⁿ diagonal, Gumbel-q, n6-only screening, continuous-q as results, per-n mixer as headline, H=256 / arch search, sum aggregation, L>8 before oracle, dense D at n≥14, mean-1/p loss, GS-index placement tricks, mismatched WS generators at new n.

Full detail: `v3/openevolve/runs/n8_phase1/opus_newfolder_scale_ideas.md`

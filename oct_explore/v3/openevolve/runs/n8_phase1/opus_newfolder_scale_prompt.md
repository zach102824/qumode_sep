You advise qumode_sep. Ideas only — no code dumps. Be concrete, ranked, and opinionated where Zach’s draft is vague.

Stack: **PyTorch cu128** on chaodao104 (2×RTX 5090). JAX is golden/spec only.

# Classical bars (lookups)
n6 8.466/16.039 | n8 13.989/29.396 | n12 32.5/76.1
(Novelty+ n8 ~13.25/28.41 — report alongside when relevant.)
Metric: Med / B90 of 1/p(GS) vs WalkSATlm LOOKUPS. Scaling success = Med(n) tracking WS growth (~Med_n12/Med_n8 ≈ 32.5/14 ≈ 2.3), not just beating Med at one size.

# Old lock (settled, too restrictive for amortization)
Circuit: vacuum | (SNAP+D → tied Cost)^L with ONE (γ, clause_w) cost unitary per formula for whole depth.
Night1 BEST n=8 L=5: Med/B90 ≈ **12.36 / 84.6** vs WS **13.99 / 29.4**. Fresh-start F1 **0/150**.
Per-instance BFGS ceiling (10 n=8, L=5, direct params, no GNN): expressivity OK — 10/10 p(GS)≥0.93, median 1/p≈1.0, B90≈1.02.
Deeper L (3/5/7/9) did not unlock B90. H=128 already used. n6↔n8 anti-corr ≈ −0.98 among strong-n6 runs under old lock.
Strong-n6 OE (held best): L=5, arch=mp, **H=128, rounds=4**, steps≈1400, lr≈1.1e−3, amax=1.8, leakpen≈0.14, tailw≈1.6 — n6 Med/B90 7.44/21.6 but n8 38.3/145.2 (bad transfer).

# Prior Opus relax-q advice (opus_relax_q_ideas.md) — USE, don’t ignore
- Free raw 2ⁿ diagonal is weak vs SNAP (both diagonal → merge); real levers: **D(α₀)** and a **structured phase head** (cost prior + per-clause 8-phase residuals), not raw phases.
- Lock: global mixer (α₀, θ, α, q̃) + **one shared** formula U_phase; train continuous q̃ → anneal to integer {1…10} at eval.
- Don’t stop Day2; don’t dual formula-conditioned diagonals / Gumbel-q / screen only on n=6.
- When GPUs free: parity → A1 primary → ceilings → warm/A2/distill.

# Zach’s NEW draft (EDIT this — do not ignore; do not invent a different goal)
Brand **new folder** (NOT continue tied-cost Day2 tree; Day2 may keep running separately — do NOT stop Day2).
1. From previous data, lock GNN architecture (H/rounds/backbone from what worked).
2. New ansatz: start with **coherent state** (free D(α₀)), not vacuum.
3. One layer = SNAP + D + **q times a fixed cost-phase unitary**; q varies by layer, q∈{1…10} for now.
4. Train at n=6, 8 and 12; test those **and larger n**; seek a **classical scaling** method (Med vs n / vs WalkSAT growth).

# Ask
Edit Zach’s 4-point plan into a concrete best plan. Be opinionated where vague:
- Is U_phase exact cost Hamiltonian or fine-tunable (cost prior + residuals)?
- Shared global q schedule vs GNN-predicted q?
- Multi-n train mix (how weighted)? Separate mixer per n?
- What GNN arch exactly to freeze from prior data?
- Folder layout and code milestones for the brand-new folder.
- Dual-GPU order, kill/promote, scaling eval protocol (which larger n: 10,14,16?).
- What NOT to do.

Structure your reply EXACTLY as:
## 1. Edited plan (Zach-facing)
## 2. Spec locks
## 3. Folder + code milestones
## 4. Dual-GPU run order
## 5. Scaling metrics / success
## 6. What NOT

Ideas only. Concrete hypers, stages, kill/promote. No code dumps.

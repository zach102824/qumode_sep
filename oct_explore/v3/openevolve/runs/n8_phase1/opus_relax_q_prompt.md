You advise qumode_sep. Ideas only — no code. Be concrete and ranked.

Stack: **PyTorch cu128** on chaodao104 (2×RTX 5090); JAX tree is golden/spec only.

# Classical bars (lookups)
n6 8.466/16.039 | n8 13.989/29.396 | n12 32.5/76.1
(Novelty+ n8 also ~13.25/28.41 — report both when relevant.)
Metric: Med / B90 of 1/p(GS) vs WalkSATlm LOOKUPS. Screen on n=8 first.

# Old lock (tested, judged too restrictive for GNN amortization)
Circuit: vacuum | (SNAP+D → tied Cost)^L with ONE (γ, clause_w) cost unitary per formula for the whole depth.
Night1 BEST n=8 L=5: Med/B90 ≈ **12.36 / 84.6** vs WalkSATlm **13.99 / 29.4**. Fresh-start F1 hit rate **0/150**.
Per-instance BFGS ceiling on the old lock (10 n=8 formulas, L=5, direct circuit params, no GNN): **expressivity OK** — 10/10 hit p(GS) ≥ 0.93, median 1/p ≈ 1.0, B90 ≈ 1.02.
Conclusion already settled: the ansatz *can* put near-unit mass on GS; the GNN Med/B90 gap is **amortization under the tied-cost lock**, not “L=5 SNAP+D can’t hit GS.” Deeper L (3/5/7/9 grid) did not unlock B90; H=128 already used; n=6↔n=8 anti-corr ≈ −0.98 among strong-n6 runs.

# NEW relaxed form (Zach GO conceptually — design the full run plan; NO “cheap tests first”)
Zach judges the fixed tied-cost phase too restrictive and wants to **relax now**. New form must be strictly more flexible than the old lock (which already “works” for per-instance BFGS).

1. **Init:** free displacement `D(α₀)`, NOT vacuum (vacuum wastes the first SNAP).
2. **Layer ℓ:** `SNAP(θ_ℓ) → D(α_ℓ) → U_phase^{q_ℓ}`
3. **`U_phase`:** free / fine-tunable diagonal phase matrix on the Gray grid, **formula-conditioned via GNN**. **Not** required to equal the true cost Hamiltonian — can be a learned fine-tune; cost-like init is optional prior only.
4. **`q_ℓ ∈ {1,…,10}`** integer so far (only 10 discrete strengths) — QAOA-adjacent, but the mixer is natural qumode SNAP+D, and the “gamma” is **less expressive** than continuous QAOA γ (discrete q only).
5. One shared `U_phase` per formula (amortized), applied `q_ℓ` times per layer.

# Context Opus already gave (Day2 post-ceiling — still a SEPARATE track; do not kill Day2)
Prior advice for the *old lock* was amortization-first (uncapped tail lse/CVaR, freeze A/TH probes, hard oversample, distill). This call is a **NEW design track** for the relaxed form. Assume Day2 may or may not be running on the farm; your plan should be for when GPUs are free / a parallel branch, not “stop Day2.”

# Ask
With all data in hand, what is the **best plan to run code** (Torch farm) for this relaxed form?

1. Updated diagnosis vs old tied lock.
2. Exact algorithm locks / free knobs (param shapes; how GNN outputs q — softmax 10-way vs round; shared vs per-layer `U_phase`?).
3. Ranked implementation + experiment plan for 2×5090 (parity vs old; n=8 first; metrics Med/B90 vs WS).
4. Ordered dual-GPU launch list (concrete hypers / stages for ~24h once GPUs are available).
5. What NOT to do.
6. Relation to QAOA: what transfers, what doesn’t, given discrete q and qumode mixer.

Structure your reply exactly as:
## 1. Diagnosis
## 2. Spec (locks / free / GNN heads)
## 3. Ranked run plan
## 4. Dual-GPU ordered list
## 5. What NOT to do
## 6. QAOA comparison (short)

Ideas only — no code dumps. Concrete hypers, stage order, kill/promote rules.

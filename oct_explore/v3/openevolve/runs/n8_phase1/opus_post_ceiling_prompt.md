You are advising on qumode_sep tied fixed-cost GNN research. Ideas only — no code. Be concrete and ranked. Stack is **PyTorch cu128** on chaodao104 (2×RTX 5090); JAX tree is golden/spec only. Existing flags: leakpen, wd, tailw, init_th_path, ema_decay, H, rounds, steps, lr, seed, lr_schedule/cosine, grad_clip, ntrain, L, bs. Algorithm locks: vacuum | (SNAP+D→cost)^L; one tied (gamma, clause_w) per formula; same L train/test; screen n=8 only; metric Med/B90 of 1/p vs WalkSATlm LOOKUPS.

# Classical bars (lookups)
n6 8.466/16.039 | n8 13.989/29.396 | n12 32.5/76.1
(Novelty+ n8 also ~13.25/28.41 — report both when relevant.)

# Night1 final (Torch tied, L=5 n=8)
Path: JAX champ 38.3 → Day1 anneal ~28 → **test Med 12.3607** (≈WS Med 13.99; selection-noise caveat).
- HOT-SWAP BEST: A_lr2e-4_tw0.5_s2 — test Med/B90 **12.36 / 84.62**
- Best test_B90 among test_Med≤13 ≈ **78** — still ≫ WS B90 29.4
- F1 fresh seed farm: **hit_rate 0/150** — kill fresh F1
- Reverse-leakpen farm: collapsed — killed
- H=128 is the working size; H=192 cosine densify was worse (Med~88 from cold/weak parents)
- Tail loss clamps exp at 3.5 (1/p≈33), so hard B90 instances get weak gradients from that term

# NEW: Per-instance BFGS expressivity ceiling (2026-10-28, box CPU, ~21s)
Script/results: `runs/n8_phase1/ceiling_bfgs_n8_L5.{py,json,md}` pushed as b6554ef5.
- NO GNN; L-BFGS-B on A/TH/g/w under tied L=5 SNAP+D→cost; 10 formulas (5 unique-GS hard + 5 random); 3 restarts × maxiter 250; NF=36.
- **All 10** reached p_succ ≥ 0.93 (most ≈1.0); median 1/p ≈ **1.00**, B90 ≈ **1.02**; leak ≤0.023.
- **Verdict: expressivity OK** — L=5 tied circuit can put near-unit mass on GS for hard n=8; GNN Med/B90 gap is **amortization / optimization**, not an ansatz wall.
- Therefore: do NOT open “deeper L / integer-repeat for expressivity” as the primary story unless amortization fails after serious B90 work.

# Known transfer issue
n=6↔n=8 anti-corr ≈ −0.98 among strong-n6 OE runs; defer n=12 until B90 moves or a growth metric (Med12/Med8 ≈ WS ~2.3) looks healthy.

# Ask
1) With expressivity cleared, what should we **actually run** on 2×5090 to improve the GNN (especially B90 toward ~29–40 while keeping Med ≲14.5)?
2) Rank concrete experiments (hypers, loss/tail redesign, oversample hard/unique-GS, batch size, init parents from BEST, early-stop/eval cadence, data size). Impact × ease, ≥8.
3) Ordered dual-GPU launch list for the next ~12–24h (GPU0 / GPU1 concrete flags).
4) What NOT to do (given F1 0/150, reverse-leakpen fail, H=192 cold fail, ceiling OK).

Structure reply exactly:
## 1. Updated diagnosis
## 2. Answers
## 3. Ranked next experiments (impact × ease)
## 4. Ordered dual-GPU launch list
## 5. What NOT to do

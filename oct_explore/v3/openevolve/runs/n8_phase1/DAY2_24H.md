# Day2 ~24h plan (post-ceiling Opus) — Torch chaodao 2×5090

Generated: 2026-10-28 Asia/Shanghai. Source: `opus_post_ceiling_ideas.md` (claude-opus-5-5-xhigh).
Supersedes the pre-ceiling fork in the prior DAY2_24H (expressivity branch / full ceiling re-run).

## Settled facts (do not re-block on these)
- **Expressivity OK.** Per-instance BFGS L=5 tied ceiling (`ceiling_bfgs_n8_L5.*`, b6554ef5): all 10 formulas p≥0.93, median 1/p≈1.00, B90≈1.02. **Do not re-run full ceiling as a blocker.** Optional *restricted* ceiling (freeze A/TH, opt only γ/w) is a Phase-0 diagnostic only.
- Night1 BEST: `A_lr2e-4_tw0.5_s2` — test Med/B90 **12.36 / 84.62** (selection-noise on N=100; true Med ~13–14). Best B90 among Med≤13 ≈ **78** ≫ WS 29.4 / Novelty+ 28.4.
- F1 fresh seed 0/150; reverse-leakpen collapsed; H=192 cold Med~88 — **always init from BEST**.
- Tail clamp at 3.5 (1/p≈33) → B90 instances get dead exp-tail grads. Primary lever = uncapped normalised tail (lse/cvar) + hard oversample + (later) warm distillation.

Bars: WalkSATlm n8 **13.989 / 29.396** | Novelty+ n8 **~13.25 / 28.41**. Select on **val**, never test. Defer n=12 until B90 moves or Med12/Med8 ≈ WS ~2.3.

## BASE (every training run)
```
--init_th_path <A_lr2e-4_tw0.5_s2 weights>
--H 128 --L 5 --rounds 4 --leakpen 0.14 --wd 1e-4 --ntrain 10000
--lr 1e-4 --lr_schedule cosine --ema_decay 0.999 --grad_clip 1.0
--steps 1500   # ≈0.5× parent 3000; eval every ~5% (~75 steps)
--bs 250       # raise to 500/1000 where noted; √k lr scale when bs↑
```
Kill at 30% steps if: val Med>16 **or** val B90 >1.10× BEST val B90 **or** leak>0.05.
Promote: lowest val B90 with val Med ≤ 14.0 (Med guard 14.5 hard kill).
Pack 3–4 short runs per GPU if VRAM allows.

New flags (small code): `tail_mode {clamp,lse,cvar}`, `tail_tau`, `cvar_q`, `tail_clamp`, `hard_frac`, `distill_lambda`, `eval_every`.

## Phase 0 — hours 0–2 (diagnostics; no GNN train)

| GPU | Work |
|-----|------|
| **GPU0** | **Eval pack:** build fresh val=1024 + sealed test=1024 (disjoint). Eval BEST + top-5 Night1 ckpts (raw+EMA). Dump per-instance 1/p, leak, #GS, unique-GS, degree; bootstrap 95% CI Med/B90. Also 1024 train formulas for train/test B90 gap. |
| **GPU1** | **Amortization probes** on ~512 val (incl. BEST worst decile): (a) restricted ceiling freeze A/TH opt γ/w; (b) warm-start from BEST preds → steps to 1/p≤2 + param move; (c) constant shared params; (d) K=4/16 oracle prototypes. Then generate warm teacher targets for full train set. |

**Gate:** train≈test B90 → objective/amortization (go Phase 1 loss). train≪test → data/ntrain (Phase 2). Warm-start close → distillation strong. Restricted B90 ≫1 → shared A/TH bottleneck (note; still do not reopen L).

CPU parallel: WS + Novelty+ lookups on fresh val/sealed.

## Phase 1 — hours 2–9 (loss repair from BEST; seeds 1,2,3)

| GPU0 | GPU1 |
|------|------|
| **ctrl:** BASE `--tailw 0.5` (old clamp loss) | BASE `--tail_mode clamp --tail_clamp 5.0 --tailw 0.5` |
| BASE `--tail_mode lse --tail_tau 1.0 --tailw 0.5` | BASE `--tail_mode cvar --cvar_q 0.1 --tailw 0.5 --bs 500` |
| BASE `--tail_mode lse --tail_tau 0.5 --tailw 0.5` | same cvar with `--cvar_q 0.2 --bs 500` |
| BASE `--tail_mode lse --tail_tau 1.0 --tailw 1.0` | BASE `--tail_mode lse --tail_tau 1.0 --tailw 0.5 --hard_frac 0.3` |

If slow: drop seed 3 first, then τ=0.5 and q=0.2.
**Gate @ h9:** W1 = best 3-seed mean val B90 with val Med≤14.0; must beat ctrl outside CI. Else keep BEST as parent into Phase 2.

Fallback if new `tail_mode` not landed yet: existing-flags densify from BEST — `tw∈{0.5,1.0,1.6}` × `lr∈{6e-5,1e-4}` × `bs∈{250,500}` × seeds, steps=1500 cosine ema — but **prioritize shipping lse/cvar**; raising tw with clamp 3.5 alone is blunt for B90.

## Phase 2 — hours 9–17 (distill + oversample + bs)

| GPU0 | GPU1 |
|------|------|
| Distill from W1 (or BEST): W1 flags + `--distill_lambda {0.3,1.0}` annealed→0 by 60% steps; seeds 1–2. If Phase0 train/test gap: also W1 `--ntrain 40000`. | ~45 min regenerate teacher from W1 preds (expert-iter round 2). Then: W1 `--hard_frac 0.3`; W1 adaptive sampling (EMA ℓ); W1 `--bs 1000 --lr 2e-4` (√k). |

Teacher = per-instance opt **warm-started from current GNN** only (never cold BFGS optima). Distance wrap-aware on angles/phases.

## Phase 3 — hours 17–24 (confirm + soup + passive growth)

| GPU0 | GPU1 |
|------|------|
| Top-2 configs × 5 fresh seeds (11–15); **one** sealed-test pass. Report 5-seed mean + best + bootstrap CI vs WS 13.99/29.40 and Novelty+ 13.25/28.41. | Soup of winner’s same-parent fine-tunes (uniform). Optional rounds+1 if MP weights share across rounds. Passive n6 + n12 eval of finalists; log Med12/Med8 (WS~2.3). **Do not select on n6/n12.** |

## Morning report checklist
- Fresh-val Med/B90 for BEST + W1 + top-5 (CI); sealed once for finalists.
- Restricted-ceiling / warm-start probe table (not a re-run of full BFGS ceiling).
- Day2 best val B90 under Med≤14.5; wall_h both GPUs.
- Growth Med12/Med8 passive only.
- WS **and** Novelty+ on fresh sets.

## Do NOT
- Cold starts / fresh F1 farms / H=192 cold / reverse-leakpen.
- Change H; deepen L / integer-repeat for “expressivity” (ceiling cleared).
- Raise tailw while clamp stays at 3.5 as the B90 plan; don’t replace clamp with plain mean(1/p).
- Distill onto cold-restart BFGS optima; don’t soup independently trained seeds.
- Select on test or on n6; don’t train n12 yet.
- Report per-instance test-time opt or #GS-as-input as the headline (diagnostics/teachers only).

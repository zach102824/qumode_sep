# NON-DEPTH next experiments — 2026-10-31 ~03:20 Asia/Shanghai

**Opus 5.5 status:** `cursor-agent --model claude-opus-5-5-high` →
`ActionRequiredError: Model not available — This model provider is not supported in your region`
(box + Mac.lan exit IP = China / CN). Retries across opus-5-5-{high,high-fast,medium,xhigh}
and opus-5-thinking-high all failed. No usable supported-region proxy on box or connected Mac.
Prior Opus replies on this box (e.g. 01:14) were intermittent; currently hard-blocked.

**Ranking below:** agent ranking under that block, constrained to Zach’s idea-space +
prior Opus 5.5 greenlights (23:56 / 01:14). NOT a live Opus reply. Re-ask Opus when
exit IP is in a supported Anthropic region (VPN/TUN with non-CN egress).

Locked: L44 champ Med/B90 = 1.051/1.473, 1.113/2.263, 1.233/4.243, 1.409/9.645; b_Med≈1.033.
No L48+ pad. Uc=exp(-iγ Hc). Blind banned.

## Ranked 4 NON-DEPTH

### 1. Fair ArmC pad L16→L44 on scale1000  — CHEAP→MED, paper_only+renames_Hc sibling
**Recipe:** Linear-interp Arm C shell (γ,φ) from L=16 → L=44 (same pad used for projector depth).
Eval SpectralAnsatz densified-E on scale1000 B=1000 (nf=16, W_H). Report Med/B90 + b_Med vs
champ L44 and ArmC L16. This is the fair same-L classical-schedule baseline the paper table lacks.
**Pass:** table row written; L×Med and b_Med reported. No promote gate (ArmC is baseline).
**Kill:** if pad ArmC@L44 Med ≥ ArmC@L16 Med on every n (interp hurts) — still publish as negative.
**Wall-clock:** ~30–90 min CPU (L44 ≈2.75× L16; B=1000).
**Honesty:** ArmC still densified Hc; fair L match only.

### 2. Two-cav Kerr budget refresh at L44 + paper mega-table finalize — CHEAP, hardware_compile+paper_only
**Recipe:** Recompute pulse/Kerr account using L44 champ |γ| stats (mean/max/p90) and L44×census
abstract pulses; fold γ mod 2π note if already verified. Rewrite PAPER_TABLE with champ=L44,
ArmC L16, **ArmC L44 (from #1)**, QAOA L16, WalkSAT R=50.
**Pass:** every number traces to RESULTS/json; champ row = locked L44 values.
**Kill:** none (documentation).
**Wall-clock:** 10–20 min after #1 (or 5 min without ArmC L44 row).
**Honesty:** paper_only / hardware_compile; still renames_Hc.

### 3. Equal-shot schedule mixture for B90 tail — MED, renames_Hc
**Recipe:** Prior Opus #3: mix 3–4 existing members (L44, L40, soft-n18, LBFGS seed0) equal-shot
p_mix=mean p_k. Spearman of log inv_p at n18 first; only run if corr < 0.95.
**Pass:** n18 B90 ≤ 95% of L44 champ with Med ≤ 102%.
**Kill:** pairwise corr > 0.95 → tail intrinsic (paper sentence).
**Wall-clock:** 20–40 min.
**Honesty:** renames_Hc (same Uc family).

### 4. Clause-weight freedom @ L44 warm-start — MED, renames_Hc
**Recipe:** Prior freedom (span_w / eps_delta) discarded at L16 vs weaker champ. Re-try span_w
warm-started from L44 pad angles, short Adam on {9,12,15}, promote only if n18 B90 ↓ and
n9–15 Med ≤ 1.05×. Still projector-weighted = soft Hc, not blind.
**Pass:** n18 B90 ≤ 95% L44 with n9–15 Med ≤ 105%.
**Kill:** no beat (likely; L16 freedom already flat) → paper “weights don’t help”.
**Wall-clock:** 40–90 min.
**Honesty:** renames_Hc (weighted sum of projectors).

Deferred: cavity packing redesign (HEAVY), n21 packs (need gen + E; HEAVY), extra noise beyond
L40 joint jitter (diminishing; optional L44 σ-grid later).

## Top-2 CHEAP RIGHT NOW
1. Fair ArmC@L44 pad eval
2. Kerr refresh + paper mega-table finalize (with ArmC@L44 row when ready)

**RIGHT NOW:** launch #1; draft #2 docs in parallel; fill ArmC@L44 when eval finishes.

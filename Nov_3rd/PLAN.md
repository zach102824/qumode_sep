# PLAN — Nov_3rd (final data)

_Asia/Shanghai 2026-11-03 — Zach t2442–t2458_

**Everything under `Nov_3rd/`** — plan, scripts, settings, tests, data, QAOA angles, runs.

## Motif
\[
|0\rangle \xrightarrow{[U_c^{X}\to\mathrm{bare\,SNAP}]^{L}} W \to \text{Fock},
\quad U_c^{X}=W\,D\,W,\quad D=\mathrm{e}^{-i\gamma E},\quad W=H^{\otimes 3}.
\]
**3 bits/cavity**. Binary. QAOA-warm shared **Hamming-shell SNAP** (φ_k, k=0..3 via popcount; code `SNAP_HEAD="shell"`). Train n=6×1000, **L-BFGS only**.

Warm dual-QAOA map (locked):
\[
\gamma^{(\ell)}=\gamma^{(\ell)}_{\mathrm{QAOA}},\qquad
\phi^{(\ell)}_{j}=-\beta^{(\ell)}_{\mathrm{QAOA}}\cdot j
\quad (j=0,\ldots,7;\ \mathrm{shared\ across\ cavities}).
\]

## Locked
| knob | value |
|------|-------|
| bits/cavity | **3** |
| W / SNAP | **BinH** / **Hamming-shell** (φ_k k=0..3; shared cavities, per layer) |
| Init | **QAOA-warm** |
| Train | **n=6 × 1000**, **L-BFGS only** |
| L | **\{1,4,8,12,16,20,24,28,32\}** small→large |
| Eval n | **CORE first** (6,9,12,15) all L → then **18** → then **21**; B=1000 |
| QAOA angle init | **β=0.01, γ=−0.01**, **nstart=1** (arXiv:2208.06909 §Parameter optimization; one train) |
| QAOA convention | qaoa_ref \(U_C=e^{-i\gamma C_n}\), \(U_B=e^{-i\beta X/2}\) |

## Fair same-L QAOA refs (locked for ladder)

Canonical angles live in `qaoa_angles/L{L}/` and **must** be paper-init n6×1000
(`ensure_qaoa_paper_n6` / `qaoa_init.py`). Old n9+n12 / grown multi-start angles
are archived as `L{L}_n912_legacy` or `L{L}_legacy_nonpaper`.

`run_final` always:
1. ensures paper-n6 QAOA for that L (+ scale1000 Med/B90 refs),
2. records warm-only baseline (exact dual),
3. polishes with warm-fallback,
4. gates accepted ckpt vs same-L QAOA; STOP on UNDER.

## QAOA parameter init
Retrain / new QAOA angles under `Nov_3rd/` use **only** the Boulebnane–Montanaro paper init (arXiv:2208.06909 §Parameter optimization): every layer starts at β=0.01, γ=−0.01, with **nstart=1** (no multi-start paper-k4 / linear-ramp / random). Shared helper: `Nov_3rd/qaoa_init.py`. Convention: qaoa_ref \(U_C=\mathrm{exp}(-i\gamma C_n)\), \(U_B=\mathrm{exp}(-i\beta X/2)\).

## Gate 0 — tests (required before any train)
```bash
python -m Nov_3rd.tests.run_tests
```
Suite T1–T12: locks, local data, packing/W=H⊗3, shell warm map, motif sealed overlap, warm≈QAOA ≤2%, blockmix fidelity, no Adam, train N=1000, eval ns, vacuum+final W, L_LIST.  
**Must all PASS.** Results → `runs/TESTS.md`.

## Debug / angle finding (2026-11-03 ~15:14–15:18 Asia/Shanghai)

**Ladder STOPPED at L=1.** Circuit is dual-correct; free polish is the bug.

### Diagnosis
1. **Warm-only = QAOA (circuit OK).**  
   Eval with \(\gamma=\gamma_{\mathrm{QAOA}}\), \(\phi_j=-\beta\cdot j\) (no free polish) on scale1000 \(B=1000\) matches QAOA L=1 **exactly**:
   | \(n\) | warm Med | QAOA Med | \(R\) |
   |------:|---------:|---------:|------:|
   | 6 | 9.779 | 9.779 | **1.000** |
   | 9 | 29.279 | 29.279 | **1.000** |
   | 12 | 90.033 | 90.033 | **1.000** |
   | 15 | 289.893 | 289.9 | **1.000** |

2. **Free n=6 L-BFGS polish UNDER (do not ship polished-as-is).**  
   Best polished ckpt (seed2) vs QAOA L=1 on scale1000:
   | \(n\) | polished Med | QAOA Med | \(R\) | verdict |
   |------:|-------------:|---------:|------:|---------|
   | 9 | 29.537 | 29.279 | 1.009 | TIE |
   | 12 | 93.283 | 90.033 | 1.036 | **UNDER** |
   | 15 | 318.811 | 289.9 | 1.100 | **UNDER** |

   Drift off warm: \(\|\Delta\phi\|\approx 0.34\), \(\gamma:\,-5.45\to-5.17\).  
   All 3 polish seeds: **train Med worse than warm Med on n=6 itself**
   (warm \(\approx 9.61\) → train \(\approx 9.7\)–\(10.0\)). Polish hurts the train set.

3. **Prior context (why “beat” used to look better).**  
   - **E0** warm-only: \(R=1.000\times\) QAOA on n9/12/15.  
   - **E1** free polish on sealed **n9+n12**: beat n9/12 (\(\sim 0.98\)–\(0.99\times\)) but failed n15 (\(\sim 1.003\)–\(1.017\times\)).  
   - **E3 L=2** warm/grow: PASS \(\sim 0.97\times\).  
   Nov_3rd lock = train **n6×1000 only** + free LBFGS → transfer weaker than E1’s n9+n12 polish.

Script / log: `scripts/debug_L1_warm_vs_polished.py`, `runs/debug_L1_warm_vs_polished.log`.

### Revised angle protocol (before free polish)
1. Ensure `qaoa_angles/L{L}/angles.json` for that \(L\).
2. **Always record warm-only baseline** on scale1000 \(\{6,9,12,15,18\}\) (and a cheap n6 train-set Med). Expect \(R\approx 1.00\) vs QAOA at L=1; keep the warm Med table.
3. Optional free L-BFGS polish (seeds 0,1,2) from warm init, train n6×1000.
4. **STOP / reject polish** if any seed’s train Med \(>\) warm Med on the train distribution (n6). Keep warm ckpt instead.
5. Eval accepted ckpt on scale1000; gate vs QAOA same \(L\):
   - \(R=\mathrm{Med}/\mathrm{Med}_{QAOA}\)
   - **BEAT** if \(R\le 0.98\); **TIE** if \(0.98<R\le 1.02\); **UNDER** if \(R>1.02\)
6. If any eval \(n\) is UNDER → **STOP** ladder; debug angles (do not climb \(L\)).
7. Else write `runs/RESULTS_L{L}.md` and proceed.

**Do not ship** a polished checkpoint that lost to warm on train Med.

**Code enforcement (train_bfgs.py):** `reject_polish_for_warm(train_med, warm_med)` is applied at the end of every `train_seed` call. Mid-loop never crowns a round with `train_med > warm_med`. If polish is missing or worse, the returned state is the exact QAOA dual (φ_j=-β·j, γ=γ_QAOA), with `used_warm=True` / `rejected_polish=True` logged and written into ckpts / `train_summary_L*.json` / RESULTS.

### Opus polish strategy (Opus 5.5 medium, 2026-11-03 15:16 Asia/Shanghai)

Full reply: `explore_OCT29/runs/opus_ideas/opus_reply_20261103_1516_nov3rd_polish_plan.md`  
Short synthesis: `runs/OPUS_POLISH_PLAN.md`  
Model used: **`claude-opus-5-5-medium`** (no region block).

**One-liner:** At L=1 accept warm **TIE**; align loss with Med + stay near warm; **first** audit whether \(n=6\) gradients transfer to \(n=15\).

**Next experiments (ordered):**
1. **Gradient-transfer audit** (Idea 1) at L=1 warm — compute \(g_n\) for \(n\in\{6,9,12,15\}\); report \(\cos(g_6,g_{15})\) on \(\gamma\) / linear-\(\beta\) / non-linear \(\phi\) parts.  
   Pass: non-linear \(\cos\ge0.5\) → keep n=6. Kill: \(\cos<0.2\) → abandon n=6 for non-linear \(\phi\).
2. **Log-ratio loss + proximal nested basis** (Ideas 2+3):  
   \(\mathcal L=\mathrm{mean}_i\log(E_i(\theta)/E_i^{\mathrm{warm}})\); fit only \(\{\gamma,\beta,c_2\}\) with \(\lambda\|\theta-\theta_{\mathrm{warm}}\|^2\), \(\lambda\in\{1,0.1,0.01\}\); val n=9, gate n=15.  
   Pass: train Med ≤ warm and \(R\le1.02\) on n12/n15. Else lock L=1 warm TIE.
3. **Grow \(L\) from warm, polish new layer only** (Idea 4 / E3-style) — confirm warm=QAOA at L=4, then L=1→4.

**Keep \(n=6\)?** Yes for \(\gamma,\beta\) by default. Abandon \(n=6\) for non-linear \(\phi\) if Idea 1 kills transfer **or** Ideas 2+3 still UNDER at n15 → train n9+n12 (Idea 5), n15 sealed.

**No Hc-free projector tricks.** Prefer keeping train \(n=6\) if possible.

## Ladder protocol
See **Revised angle protocol** above. Same gate thresholds. Locks unchanged unless Opus + Zach explicitly override the train set.

## Run
```bash
source /workspace/qtimes_venv/bin/activate
export PYTHONPATH=/workspace/qumode_sep
python -m Nov_3rd.tests.run_tests && python -m Nov_3rd.run_final
```

Debug warm vs polished:
```bash
python -u Nov_3rd/scripts/debug_L1_warm_vs_polished.py --ns 6 9 12 15
```

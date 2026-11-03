# LIVE Opus 5.5 high — SNAP QAOA-warm explore plan

**Status:** LIVE OPUS SUCCEEDED (`claude-opus-5-5-high`, ask mode)  
**When:** Asia/Shanghai 2026-11-03 ~12:11–12:18  
**Brief:** `explore_OCT29/runs/opus_ideas/brief_20261103_1211_snap_qaoa_warm_plan.txt`  
**Raw:** `explore_OCT29/runs/opus_ideas/opus_reply_20261103_1211_snap_qaoa_warm_plan.md`  
**Zach:** t2412 — execute nonstop, no confirm.

---

## One-liner

Every packing starts as the identical QAOA circuit, so we screen bits-per-cavity \(b\in\{1,2,3,4,6,12\}\) at n=12 with QAOA-warm gauss/shell/free SNAP at \(L\le8\), then carry the winner to n18 transfer and L=16/32 against a new QAOA L=32 baseline. Bin vs Gray is an exact identity in ideal simulation (it only matters for Kerr level labeling and photon loss), and free θ should land close to shells because variables inside a cavity are interchangeable.

---

## Diagnosis

1. **Packing is the axis that matters.** At fixed n every packing starts as the same QAOA circuit; post-train differences come only from extra mixer freedom.
2. **Bin vs Gray is an identity in ideal sim** (Π cancels in the logical frame). Gaps in the 10-30 BinH/GrayH run are training-path noise. Frame matters for Fock-native heads (Kerr) under level relabel and for photon-loss noise.
3. **Free θ ≈ shells.** Random 4-SAT is invariant under bit permutations inside a cavity → optimum tends to popcount shells. Free θ only adds symmetry-breaking.
4. **QAOA-warm guarantee:** heads that contain the QAOA point (x_pop, bits, gauss, shell, free) start exactly at QAOA; train loss ≤ QAOA if best checkpoint kept. Zero-shot n can slip (E1: −0.3–1.7% at n15).
5. **Gain grows with depth.** ~1–4% at L=1–2, ~40% at L=16 → packing differences invisible at L≤2; screen at L=4–8, decide at L≥16.
6. **Best b expected mid-range (~3–6).** At b=n shells give global symmetric mixers that plateau.
7. **Code:** `packing.py` hard-codes K=3; need bit-frame block-mixer for generic b (exact under ideal W, \(2^n\) not \(nf^C\)).

Mixer Walsh form (\(W\) ideal):
\[
M(\theta)=H^{\otimes b}\mathrm{diag}(e^{-i\theta})H^{\otimes b}=\exp\Big(-i\sum_{S\subseteq[b]}\hat\theta_S X^S\Big).
\]
QAOA point: \(\theta_B=-\beta\,\mathrm{pc}(B)\) → \(M=e^{-i(\beta/2)\sum X}\).

---

## Ranked ladder

| # | Title | Pass / kill | Packing & L | Wall | Label |
|---|---|---|---|---|---|
| **0** | Bit-frame blockmix + identity + QAOA L=4/8/32 baselines | Med within 0.01% on IDs; miss → stop | b∈{1..12}; L∈{1,2,4,8,16,32} | 1–1.5 h | [IDENTITY] |
| **1** | **PACK-12 screen** (x_pop/gauss/shell/free, 3 seeds) | Pass: best (b,head) \(R_8\le0.95\times\) b=3 same head on ≥2/3 seeds. Kill: all b within 2% | matrix below; L∈{2,4,8} | 2–3 h | [EXACT-WARM] |
| **2** | Head ladder b=3 L=16 | Beat QAOA L=16 + Arm C; stretch=Nov progressive shell. Kill free if ≱2% better than shell | b=3 L=16 | 3–4 h | [EXACT-WARM] |
| **3** | Transfer-safe L=16 polish (close E4 3–5% gap) | ≤ Arm C champ n9/12/15 then n18. Kill: >2% behind after 2 seeds | b=3 L=16 | 2–3 h | [EXACT-WARM] |
| **4** | Packing transfer top-2 | Winning b lowers b_Med by ≥0.01 vs b=3 and R(18) by ≥5% | b∈{1,2,3,6(+winner)}; L∈{8,16} | 3–5 h | [EXACT-WARM] |
| **5** | Depth-matched L=32 | Report \(R_{32}\); bonus if dual-beats sealed | best b, L=32 | 4–6 h | [EXACT-WARM] |
| **6** | Kerr popcount-sorted relabel (gated ≤2p reopen) | Reopen only if ≤1.20× sealed n9+n12; else kill ≤1 h | b=3 L=16 | 1 h | [PROJ-WARM] |
| **7** | Clause-aware var→cavity assignment | ≥3% Med else kill | b=3 n12 L=8 | 1–2 h | [PREPROC] |
| **8** | Per-cavity θ kill | Expect ≈ shared; kill if within 1% | b=3 n12 L=8 | 0.5 h | [EXACT-WARM] |
| **9** | Bin vs Gray under photon loss | Keep Gray for HW story only if degrades less | b=3 L=8 | 1–2 h | [NOISY] |
| **10** | Strict vs hybrid warm at depth | Report diff; label which any result used | best b | 2–4 h | [STRICT]/[HYBRID] |

---

## Packing matrix (n=12, L∈{2,4,8})

| b×C | Shell | Gauss | Free | XX pairs (of 66) | Hardware |
|---|---|---|---|---|---|
| 1×12 | 1 (=QAOA) | — | 1 | 0 | qubit control |
| 2×6 | 2 | =shell | 3 | 6 | easy |
| **3×4** | 3 | 2 | 7 | 12 | current |
| 4×3 | 4 | 2 | 15 | 18 | plausible |
| 6×2 | 6 | 2 | 63 | 30 | stretch |
| 12×1 | 12 | 2 | 4095 | 66 | ideal only |

---

## First 6–12h stack (executing now)

1. **0–1.5 h:** item 0 — blockmix, identity checks, QAOA L=4/8/32 baselines. Gate: nothing else until pass.
2. **1.5–4.5 h:** item 1 PACK-12 + item 8.
3. **4.5–8 h:** item 4 transfer + item 2 head ladder L=16.
4. **8–11 h:** item 5 L=32 + item 3 polish.
5. **11–12 h:** write-up; gated 6/7 if slack.
6. **Ping Zach only if:** (a) b≠3 ≥5% at n18 transfer, (b) item 6 reopens ≤2p, (c) L=32 dual-beats sealed.

Protocol lock: train n9=200 + n12=128; eval scale1000 B=1000; always QAOA-warm \(\theta=-\beta\cdot\mathrm{pc}\), \(\gamma=\gamma_{\mathrm{QAOA}}\).

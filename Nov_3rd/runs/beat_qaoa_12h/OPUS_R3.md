# Opus R3 — shallow n15 after global-search evidence (2026-11-03 ~21:38 Asia/Shanghai)

**Call:** succeeded · model `claude-opus-5-5-medium` · ask mode · exit 0  
**Sources:** `explore_OCT29/runs/opus_ideas/brief_20261103_2136_shallow_r3.txt` · raw `opus_reply_20261103_2136_shallow_r3.md`

## #1 Recommendation: declare shallow n15 capacity-limited under Hamming-shell

Remaining EV of more shallow-n15 training ~10–15% for clean eval ≤0.98. Prefer write-up of L=16 clear beat + shallow n12.

### Why close to proven
- All methods land eval n15 in 0.983–0.997.
- L=1 (0.984) ≈ best L=2 (0.983) — extra layer buys ~0.001.
- Global search train R=0.9807 → eval 0.997 (overfit train pack; not ≤0.978 reopen).
- Gap 0.983→0.980 inside scale1000 SE; repeated eval looks = selection on noise.

### Closing check (~1–1.5h) — DO THIS BEFORE DECLARE
Same global search on a **large n15 pack disjoint from train AND scale1000** (500–1000).
- best R **>0.980** → capacity proven; stop & declare (most likely).
- best R **≤0.978** → reopen → #2.

## #2 (only if reopen): retrain dense pack + held-out early stop
Warm free-φ+γ from 0.983 / dense optimum; never early-stop on scale1000. Kill if val R>0.981 after ~1h. One scale1000 shot only if val≤0.979. Wall 2–3h.

## #3: match eval metric + mild CVaR tail
Exact eval statistic as loss; CVaR τ≈0.2–0.3 weight ~0.3; multi-n n12+n15. Kill if val>0.981 or n12>0.978. Wall 1.5–2h.

## Bottom line
Run closing check, then most likely declare. Any eventual shallow-n15 "beat" needs held-out val + single eval shot.

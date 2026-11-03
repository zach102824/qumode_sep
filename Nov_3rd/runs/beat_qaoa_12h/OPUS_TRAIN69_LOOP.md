# OPUS TRAIN{6,9} LOOP — Round 1 (2026-11-03 ~22:40 Asia/Shanghai)

**Lock (t2563u):** train ONLY {6,9}; eval CORE {6,9,12,15} R≤0.98; L∈{2,16}; gates locked.
**Idea-gen:** R1 = Opus `claude-opus-5-5-medium`. **R2+ = Grok `cursor-grok-4.6-high`.**

Brief: `explore_OCT29/runs/opus_ideas/brief_20261103_2240_train69_loop.txt`
Reply: `explore_OCT29/runs/opus_ideas/opus_reply_20261103_2240_train69_loop.md`

## Opus bottom line
- L=16 under train{6,9}: **likely clear-beat** all CORE (esp. with n9-weighted unscaled head).
- L=2 n15 ≤0.98: **not expected** under this head (dense ceiling 0.9803 even with n15 in train).

## Ranked ideas
1. **ratio69_w9** — unscaled shared γ,c1–c3; w6=1,w9=2; early-stop on held-out n9; Adam→LBFGS. Run L=16 seeds 0–5 FIRST.
2. **power_p** — a_n=(8/(n−1))^p; learn p (init 0) + fixed-p hedge {−0.5,0,0.5,1}; penalize scaled c2,c3.
3. **trust69_softmed** — freeze γ,c1; ball ρ on c2,c3; soft-median loss; best-effort shallow.

## Run order
1. Idea1 L=16 s0–5
2. Idea2 L=16
3. L=2 Idea1+3 on spare cores

# net16x — design note (Zach, 2026-10-22)

Fresh non-stop OpenEvolve run attacking the hard n=16 tail with three new mechanisms, built on ONE new gate type.
net16r is paused (STOP file), untouched. Baselines: QAOA L4 34.82/98.20; best so far net16r ckpt_10 83.9/2607 (fair TEB, rr median/B90 of 1/p).

## Locked (unchanged)
Task (train n=8 [2 cavities] + n=12 [3 cavities], 16 levels; select VAB 60 n=16; fair test TEB 100 n=16), fitness (fair round-robin
-(ln med + ln B90)/2; loss mean+median+CVaR), anti-cheat pv_acc>0.85 → −20, prep, readout, compiler and simulator core
(frozen_circuit.py / evaluator_net.py, hash-verified on every evaluation). No global-cost imitation.

## Step 1 — failure analysis → `FAILURE_ANALYSIS_16x.md`, prompt summary `runs/net16x/FAILURE_SUMMARY_16x.txt`
Tail = unique-ground-state formulas; pv points to a wrong basin (pv_acc 0.49); amplitude spreads into low-energy false basins ~8 flips
from the solution; layers 3–4 add nothing on the tail; chain geometry does not discriminate.

## Step 2 — the new gate: neighbour pair unitary (`net/pair_circuit.py`, tests `net/test_pair16x.py`)
U = exp(−iH), any Hermitian 256×256 H on cavities (a, a+1), joint index 16·i_a + i_b. Three slots per layer: 0 before the cost phase,
1 between cost phase and mixer, 2 after the mixer (≤ 3·(C−1) per layer). Program outputs: `pslots`; shared `pHr/pHi` (L,S,A,256,256),
A = C−1 or 1 (same on every pair); or structured per-formula `pG/pGi` (K≤16 basis) + `pc` (F,R,L,S,C−1,K). The evaluator Hermitises,
exponentiates (JAX Pade expm, differentiable) and applies by einsum on the (B,16,16,16,16) chain state. `circuit_px` == frozen `circuit_p`
when no pair keys are given (bitwise, tested at n=8/12/16). `n_pair_unitaries` (nonzero blocks per circuit) is reported in metrics
(stage 1, stage 2, TEB check). Guard: 256×256 shapes now allowed; still forbidden: 1<<n / 2**n / 65536 / 4096 tables, itertools.product
over bits, argmin, solution / ground-state lookup.

## Step 3 — islands (protected for the first 160 iterations = 40 evaluations per island)
| island | mandate | seed (`net/seeds16x/`) |
|---|---|---|
| 0 reencode | per-layer bit-to-cavity relabel by pair permutation unitaries conjugating the mixer (slot1 P, slot2 P†), 48 generators (π/2)(I−P_k): CNOT relabels A→B, B→A, bit SWAPs | net16r ckpt_10 body + head |
| 1 grover | Grover-like reflections: clause-rank local oracle (π on capped best levels) + pair diffusion exp(−iθπ|s⟩⟨s|) and per-cavity reflections, slot 2 | net16r ckpt_110 (grover family) body + head |
| 2 nondiag | non-diagonal pair mixer after the in-cavity mixer: XX and YY (flip-flop, beam-splitter-like) correlated cross-boundary generators, slot 2 | net16r ckpt_10 body + head |
| 3 wildcard | combinations; opens at iteration 160 (`runs/net16x/wildcard.open`), before that rotates the three mandates | copy of best seed |
All heads are zero-initialised (seed == parent at init), shared over pairs (size-free; ≤8 expm per training step). Opus 5.5 medium reviewed the
seed design (`runs/net16x/opus_seed_design.txt`): bosonic a†b dropped from nondiag (photon-number conserving, mixes bits through carries),
exact bit SWAPs added to reencode. During protection: migration off (`config_net16x_protected.yaml`), prompts carry the island mandate and
forbid reverting to net16r families; reseeding disabled until wildcard opens and then keeps each island's direction name.

## Step 4 — machinery (copies of net16r)
`cursor_adapter16x.py` (:8766, Grok 4.6 high-fast code, Opus 5.5 medium idea every 5th call, retry/backoff, prompt file for long prompts,
framing + failure summary + gate doc + COMPUTE & NOVELTY + mandate), `net/evaluator_net16x.py` (3-stage cascade, freeze-aware clock,
s1_margin 0.20 file, XLA cache runs/jaxcache16x), `net16x_driver.sh` (protected → explore 60 / exploit 40 forever, crash back-off),
`net16x_watchdog.sh`, `net16x_services.sh`, `eval_governor16x.py` (GOV_SLOTS=2 FLOOR=0 NEED=0), `net16x_checker.py` + `net/chunked_check16x.py`
(fair TEB every 10 iterations → `runs/net16x/check/checkpoint_k.json`, same fields as net16r + n_pair_unitaries), `reseed16x.py`,
`bootstrap16x.py`, `env16x.sh`. Stop: `touch runs/net16x/net16x.STOP`.

## Caveat
A pair unitary with arbitrary H is a hardware assumption (not yet shown native); a winner must be compiled and its cost compared fairly with QAOA's ZZZZ terms.

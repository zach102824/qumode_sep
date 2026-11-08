# Cost reconciliation, audit 1

No optimizer ran. Forty-eight fixed-state forwards recomputed the stored L5 vectors. Those forwards are diagnostic and are not in the historical ledgers below.

## 384: published 124788 versus 53927 + 70477 = 124404

RECONCILED.

`run_meta.json` at `7e2183b7` stores `evals_new` = 70477 and `family_inclusive` = 124788. The inclusive figure adds the fresh marginal, the descending marginal, and the frozen source constant 53927.

Each finished start in `fresh_L*.json` and `descend_L*.json` stores `nfev` equal to the L-BFGS trace length plus one. The extra one is the forward after the optimizer returns. `evals_new` counts only the trace calls. There are 288 fresh starts and 96 descending starts, 384 finished starts, and zero censored starts. Sum of stored `nfev` is 70861. Subtracting 384 leaves 70477, which matches `evals_new`.

124788 − 124404 = 384. The published inclusive total includes those 384 finish forwards. Adding 53927 to `evals_new` leaves them out. No start is missing, and no evaluation was invented.

Per-evaluation IDs inside each L-BFGS trace were not logged. The stored event is the start, keyed by `(7e2183b7, arm, depth, chain k)`.

## 1000: source 53927 versus 52927, and 28099 versus 27099

RECONCILED. The 1000 is not a recorded batch.

The round-5 protocol hardcoded the arm-2 source cost as 28099. The cells that constant claims to add are still in the tree at `d9af7f2a`:

| source | nfev_sum |
|---|---|
| round-4 plain L1 | 575 |
| round-4 plain L2 | 1248 |
| round-4 plain L3 | 2353 |
| round-4 plain L4 | 8900 |
| round-4 subtotal | 13076 |
| round-3 A_sub_M L1 | 322 |
| L2 | 622 |
| L3 | 1699 |
| L4 | 3471 |
| L5 | 3723 |
| L6 | 4186 |
| round-3 subtotal | 14023 |
| raw total | 27099 |

28099 − 27099 = 1000. There is no event whose `nfev` is 1000. The round-3 six-term sum is 14023. Writing that subtotal as 15023 produces exactly this 1000. Round-5 arm 2 then added its own marginal 25828 to the hardcoded 28099 and published 53927. Using the raw cell sum instead gives 25828 + 27099 = 52927.

Frozen v1 remains 53927. This audit does not replace it. Issued rescue allowances that used 53927 are unchanged. For the completed p11 and p12 budgets, `4 * Rp` is larger than both 52927 and 53927, so neither constant is the binding cap there.

## v2 amortization

Fresh marginal, including the finish forwards, is 48488. Descending marginal is 22373. Raw upstream source, counted once, is 27099. Family inclusive on that raw convention is 97960. The published family inclusive 124788 remains the v1 arithmetic: those marginals plus the frozen 53927.

Per-chain sums through L5 and through L6 are in `cost_ledger_v2.json`. Round-6 records do not contain an optimizer-success flag. Censored starts are zero. Starts with a poor energy are a tail of completed optimizations, not missing logs.

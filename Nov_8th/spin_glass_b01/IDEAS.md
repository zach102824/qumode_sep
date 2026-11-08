# Historical B=0.1 round: confirm the |J|-max packing

This round is the historical author benchmark `B = 0.1`. It is not a critical-field run. `B_cri` stays withheld. No new field rule is introduced.

## Why this change

Round 4, on development instances k=4..15, kept the Nov 3 shared-gamma circuit and changed only the cavity assignment. The triples that maximize the sum of `|J|` inside each triple, with ties broken lexicographically, captured more absolute coupling than the fixed interleaved triples on 71 of 72 round-3 instances. At L=6 that packing had median fidelity differences versus equal-depth QAOA of +0.219, +0.056, and +0.072 at N=6, 9, and 12, against +0.116, +0.029, and −0.020 for the interleaved KM reference. It also gave the round-3 failure N9 k7 a nonzero ground-space weight. The parameter count stayed 3+3L.

That development result was not a confirmation. Per-layer gamma, which the same round also fit, is out of scope and is not run. The QAOA-projection initialization from round 6 did not lower the stored interleaved energy and is not repeated.

## What is fit

Fresh instances k=44..55. These indices were not used in rounds 3, 4, 5, or 6. The `J` law and `B = 0.1` are the historical ones. Families are the nested shared-gamma ladder on the interleaved packing, the same ladder on the `|J|`-max packing, and equal-depth QAOA at the round-3 budget. QAOA has no triples, so its circuit is unchanged.

The primary comparison is KM_pack against that QAOA at the same depth. Interleaved KM is the reference on the same instances. Equal parameter count, KM_pack at L=3 against QAOA at L=6, is reported and is not the verdict.

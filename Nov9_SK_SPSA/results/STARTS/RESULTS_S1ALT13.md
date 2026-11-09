# STARTS S1ALT13: 1:3 shell-heavy vs PR88 1:1 at N15 L=2 Q=2048

Exclusive unread seeds 10104–10111. Frozen S1JOINT and exact PR88 S1ALT11. Energy selects. F post hoc.
S1ALT13: k%4==0 shared Uc/γ, else SNAP/β. Leftover continue_phase ABB. K=1019 → 255/764. GR not patched.

Runs 48 ok 48 failures 0. Charged 98304. Outside cap 0. Rejected B* 0. Missing 0.

Gate N15 L2 Nov3 mean ΔE (S1ALT13−S1ALT11) = +0.001462 med +0.000924 CI [-0.000224, +0.003578] n=8: **alt13_does_not_clear_alt11**.

## Pairwise energy (first minus second)

| circuit | contrast | Δ(E−E0)/N mean/med [CI] tails wins | ΔE | ΔF wins |
|---|---|---|---|---|
| Nov3 | S1ALT13_minus_S1ALT11 | +0.00010 med +0.00006 [-0.00001,+0.00024] tails [-0.00015,+0.00052] 2/8 | +0.00146 med +0.00092 [-0.00022,+0.00358] tails [-0.00218,+0.00785] | -0.00181 med -0.00163 [-0.00265,-0.00106] tails [-0.00418,-0.00020] 0/8 |
| Nov3 | S1ALT13_minus_S1JOINT | -0.00082 med -0.00089 [-0.00127,-0.00032] tails [-0.00160,+0.00047] 7/8 | -0.01231 med -0.01340 [-0.01902,-0.00485] tails [-0.02394,+0.00710] | -0.01532 med -0.01419 [-0.02055,-0.01019] tails [-0.02560,-0.00362] 0/8 |
| Nov3 | S1ALT11_minus_S1JOINT | -0.00092 med -0.00092 [-0.00130,-0.00053] tails [-0.00166,-0.00005] 8/8 | -0.01377 med -0.01381 [-0.01954,-0.00795] tails [-0.02486,-0.00075] | -0.01351 med -0.01304 [-0.01813,-0.00882] tails [-0.02300,-0.00342] 0/8 |
| QAOA | S1ALT13_minus_S1ALT11 | -0.00000 med -0.00000 [-0.00000,-0.00000] tails [-0.00000,+0.00000] 4/8 | -0.00002 med -0.00001 [-0.00004,-0.00000] tails [-0.00006,+0.00001] | +0.00036 med +0.00032 [+0.00010,+0.00064] tails [-0.00006,+0.00097] 5/8 |
| QAOA | S1ALT13_minus_S1JOINT | -0.00029 med -0.00029 [-0.00033,-0.00025] tails [-0.00039,-0.00020] 8/8 | -0.00432 med -0.00436 [-0.00492,-0.00372] tails [-0.00583,-0.00293] | +0.00930 med +0.00876 [+0.00796,+0.01075] tails [+0.00674,+0.01265] 8/8 |
| QAOA | S1ALT11_minus_S1JOINT | -0.00029 med -0.00029 [-0.00033,-0.00025] tails [-0.00039,-0.00019] 8/8 | -0.00430 med -0.00434 [-0.00490,-0.00371] tails [-0.00584,-0.00290] | +0.00894 med +0.00871 [+0.00773,+0.01015] tails [+0.00620,+0.01169] 8/8 |

## Fair Nov3 vs QAOA (not the 1:3-vs-1:1 gate)

Energy-win candidate only if that CI lies entirely below 0.

| arm | Δ(E−E0)/N Nov3−QAOA mean/med [CI] tails Nov3 lower | ΔF | energy-win candidate? |
|---|---|---|---|
| S1JOINT | +0.00315 med +0.00312 [+0.00230,+0.00402] tails [+0.00132,+0.00519] 0/8 | +0.00761 med +0.00868 [-0.00104,+0.01635] tails [-0.01325,+0.02660] 6/8 | no |
| S1ALT11 | +0.00252 med +0.00222 [+0.00164,+0.00347] tails [+0.00099,+0.00491] 0/8 | -0.01484 med -0.01596 [-0.02289,-0.00664] tails [-0.03213,+0.00309] 1/8 | no |
| S1ALT13 | +0.00262 med +0.00213 [+0.00167,+0.00369] tails [+0.00103,+0.00544] 0/8 | -0.01702 med -0.01738 [-0.02561,-0.00831] tails [-0.03534,+0.00178] 1/8 | no |

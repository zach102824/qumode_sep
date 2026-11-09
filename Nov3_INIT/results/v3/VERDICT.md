# N3INIT v3 verdict

`spread8` keeps the coherent Nov3 center and replaces four SPSA blocks with eight shorter ones. Both methods pay the new split. Train mean Nov3 energy is worse than `center4` by +0.292, +0.048, and +0.012 at `q=128`, `512`, and `1024`. The gap versus QAOA also gets worse. The rule keeps `center4` at every `q`. The test half moves the same way: every reported cell has higher Nov3 energy under `spread8` than under `center4`. At `q=128` the short blocks (`T=2` at `L=1`) give back the low-budget gains. Failures: 0. New queries: 106496 per half.

The allocation result on this panel is stable. Four blocks with the coherent start beat eight short blocks. The 8-to-2 halving from v1 is not a better default once the center is separated.

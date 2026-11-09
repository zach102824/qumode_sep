# N3INIT v2 verdict

`center4` changes only Nov3 start 0, to `γ=0.25` and `(s1,s2,s3)=(-0.5,0,0)` on every layer. Starts 1–3 reproduce v1 `equal4` within `1e-8` on all 96 cells. QAOA was not rerun. New Nov3 queries: 53248 per half. Failures: 0. Cell time about 42s per half.

The train rule adopts `center4` at `q=128`, `512`, and `1024`. The test half, a second look, adopts it at all three as well. The energy drop versus `equal4` is −0.100 on train and −0.179 on test at `q=128`, and about −0.01 to −0.02 at the higher budgets. The Nov3−QAOA energy gap improves at every `q` on both halves. At `q=128` the test gap falls from +0.203 to +0.025. At `L=1` and `q≥512` the selected start is usually not start 0, so the two arms match and the 4/4 energy wins against QAOA remain.

Low budget and `L=3` are where the center moves the result. On the test half at `q=128`, `N=12`, `L=3`, energy wins go from 0/4 to 3/4. `N=12`, `L=1`, `q=128` improves by 0.36 in mean energy and still loses 0/4 to QAOA.

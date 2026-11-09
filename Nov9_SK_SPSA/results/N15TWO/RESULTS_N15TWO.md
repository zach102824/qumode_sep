# N15TWO: N=15 L=1 two-template bimodal cluster-mean transfer

Unmodified v4 `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / PR79 `347ac23369db`.
Train 10760–10767, test 10768–10775. N=15 L=1 Q=1024 T=122 center-4. Energy selects. F post hoc.
Two Nov3 cluster means (sign(s1)) and two QAOA cluster means (sign(beta)) frozen from train before test.
Two-template energy = min of nonempty cluster-template energies. Unsigned/sign-aligned secondary only. No GR. Does not reopen PR119.

Fits 32 ok 32. Transfer evals 56 ok 56. Charged 32824. Rejected B* 0.
Nov3 templates 2 (n_pos=4 n_neg=4); QAOA templates 2 (n_pos=1 n_neg=7).

**Label TWO_TEMPLATE_TRANSFER_NEGATIVE_CI.**
primary two-template Nov3 − two-template QAOA: -0.38781 med -0.36948 [-0.45689,-0.33312] tails [-0.60076,-0.25563] 8/8
1 train independent Nov3−QAOA: -0.33139 med -0.29441 [-0.39784,-0.27285] tails [-0.50611,-0.24746] 8/8
2 test ceiling independent Nov3−QAOA: -0.32391 med -0.32576 [-0.38799,-0.26529] tails [-0.49862,-0.19834] 8/8
3 two-template Nov3 vs independent QAOA: -0.30839 med -0.32666 [-0.36412,-0.25395] tails [-0.44468,-0.18425] 8/8
4 transfer gap two-template Nov3 vs independent Nov3: +0.01552 med +0.00637 [-0.01589,+0.04610] tails [-0.06519,+0.09905] 3/8
5a Nov3 cluster+ vs unsigned QAOA: -0.58413 med -0.58709 [-0.81380,-0.38191] tails [-1.23192,-0.13521] 8/8
5b Nov3 cluster− vs unsigned QAOA: -0.61434 med -0.57421 [-0.75249,-0.50325] tails [-1.04235,-0.38288] 8/8
5c Nov3 cluster+ vs independent QAOA: -0.22960 med -0.23233 [-0.32958,-0.11942] tails [-0.44468,+0.06342] 7/8
5d Nov3 cluster− vs independent QAOA: -0.25980 med -0.28540 [-0.31675,-0.18912] tails [-0.35458,-0.05242] 8/8
6a unsigned-mean Nov3 − unsigned QAOA (replication): +1.65928 med +1.67016 [+1.57076,+1.74135] tails [+1.43733,+1.86629] 0/8
6b sign-aligned Nov3 − unsigned QAOA (replication): +1.32844 med +1.35138 [+1.23233,+1.41948] tails [+1.08227,+1.53836] 0/8

Train Nov3 unsigned n=8; gamma mean +0.0250 std 0.3310 range [-0.2999,+0.4400]; theta_K mean -0.3832 std 0.0458 range [-0.4460,-0.3184]; theta_M mean -0.0047 std 0.0199 range [-0.0336,+0.0203]; s1 mean +0.0176 std 0.9270 range [-0.8805,+0.9157]; s2 mean -0.0583 std 0.0390 range [-0.1337,+0.0032]; s3 mean +0.0902 std 0.2097 range [-0.1689,+0.5165]
Train Nov3 sign-aligned n=8; gamma mean +0.0250 std 0.3310 range [-0.2999,+0.4400]; theta_K mean -0.3832 std 0.0458 range [-0.4460,-0.3184]; theta_M mean -0.0047 std 0.0199 range [-0.0336,+0.0203]; s1 mean +0.8669 std 0.0283 range [+0.8241,+0.9157]; s2 mean -0.0583 std 0.0390 range [-0.1337,+0.0032]; s3 mean +0.0902 std 0.2097 range [-0.1689,+0.5165]
Train Nov3 cluster+ n=4; gamma mean +0.3309 std 0.0769 range [+0.2649,+0.4400]; theta_K mean -0.3901 std 0.0516 range [-0.4460,-0.3368]; theta_M mean -0.0146 std 0.0245 range [-0.0336,+0.0203]; s1 mean +0.8845 std 0.0225 range [+0.8620,+0.9157]; s2 mean -0.0764 std 0.0411 range [-0.1337,-0.0358]; s3 mean +0.1498 std 0.2482 range [-0.0075,+0.5165]
Train Nov3 cluster- n=4; gamma mean -0.2810 std 0.0127 range [-0.2999,-0.2737]; theta_K mean -0.3762 std 0.0459 range [-0.4216,-0.3184]; theta_M mean +0.0052 std 0.0078 range [-0.0031,+0.0122]; s1 mean -0.8493 std 0.0233 range [-0.8805,-0.8241]; s2 mean -0.0402 std 0.0314 range [-0.0679,+0.0032]; s3 mean +0.0306 std 0.1776 range [-0.1689,+0.1899]
Train QAOA unsigned n=8; gamma mean +0.2083 std 0.2085 range [-0.2897,+0.4136]; beta mean -0.2933 std 0.2745 range [-0.3947,+0.3861]
Train QAOA cluster+ n=1; gamma mean -0.2897 std 0.0000 range [-0.2897,-0.2897]; beta mean +0.3861 std 0.0000 range [+0.3861,+0.3861]
Train QAOA cluster- n=7; gamma mean +0.2794 std 0.0593 range [+0.2508,+0.4136]; beta mean -0.3904 std 0.0037 range [-0.3947,-0.3852]

B* wall 18.491856336593628. Train fit wall 8.03872299194336. Test wall 8.26843786239624. Sum cell wall 54.8s. Peak RSS KiB 190680.

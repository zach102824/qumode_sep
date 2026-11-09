# N12TWO: N=12 L=1 two-template bimodal cluster-mean transfer

Unmodified v4 `de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1` / PR79 `347ac23369db`.
Train 10792–10799, test 10800–10807. N=12 L=1 Q=1024 T=122 center-4. Energy selects. F post hoc.
Two Nov3 cluster means (sign(s1)) and two QAOA cluster means (sign(beta)) frozen from train before test.
Two-template energy = min of nonempty cluster-template energies. Unsigned/sign-aligned secondary only. No GR. Does not reopen PR119.

Fits 32 ok 32. Transfer evals 56 ok 56. Charged 32824. Rejected B* 0.
Nov3 templates 2 (n_pos=5 n_neg=3); QAOA templates 2 (n_pos=5 n_neg=3).

**Label TWO_TEMPLATE_TRANSFER_NEGATIVE_CI.**
primary two-template Nov3 − two-template QAOA: -0.30564 med -0.29873 [-0.38337,-0.24151] tails [-0.53162,-0.14831] 8/8
1 train independent Nov3−QAOA: -0.26717 med -0.24691 [-0.31196,-0.22288] tails [-0.37811,-0.17518] 8/8
2 test ceiling independent Nov3−QAOA: -0.28623 med -0.27855 [-0.33533,-0.24146] tails [-0.42894,-0.18583] 8/8
3 two-template Nov3 vs independent QAOA: -0.25857 med -0.27349 [-0.30110,-0.20724] tails [-0.34835,-0.11772] 8/8
4 transfer gap two-template Nov3 vs independent Nov3: +0.02766 med +0.02202 [+0.00626,+0.05000] tails [-0.02013,+0.08059] 2/8 (CI entirely >0: True; descriptive only; does not switch primary)
5a Nov3 cluster+ vs unsigned QAOA: -1.69428 med -1.64347 [-1.95721,-1.46046] tails [-2.40603,-1.11851] 8/8
5b Nov3 cluster− vs unsigned QAOA: -1.71287 med -1.64786 [-1.91965,-1.52525] tails [-2.23910,-1.32035] 8/8
5c Nov3 cluster+ vs independent QAOA: -0.20485 med -0.23661 [-0.28286,-0.11026] tails [-0.34835,+0.08412] 7/8
5d Nov3 cluster− vs independent QAOA: -0.22344 med -0.23463 [-0.26618,-0.17736] tails [-0.30284,-0.11772] 8/8
6a unsigned-mean Nov3 − unsigned QAOA (replication): -0.07891 med -0.07341 [-0.09744,-0.06330] tails [-0.12811,-0.04129] 8/8
6b sign-aligned Nov3 − unsigned QAOA (replication): -0.76525 med -0.78205 [-0.83596,-0.69521] tails [-0.96126,-0.56616] 8/8

Train Nov3 unsigned n=8; gamma mean +0.0958 std 0.3189 range [-0.3398,+0.3760]; theta_K mean -0.3790 std 0.0420 range [-0.4224,-0.2996]; theta_M mean +0.0140 std 0.0487 range [-0.0207,+0.1265]; s1 mean +0.2253 std 0.8985 range [-0.8798,+0.9127]; s2 mean -0.0031 std 0.0693 range [-0.1477,+0.0875]; s3 mean -0.1193 std 0.1878 range [-0.3534,+0.1300]
Train Nov3 sign-aligned n=8; gamma mean +0.0958 std 0.3189 range [-0.3398,+0.3760]; theta_K mean -0.3790 std 0.0420 range [-0.4224,-0.2996]; theta_M mean +0.0140 std 0.0487 range [-0.0207,+0.1265]; s1 mean +0.8697 std 0.0279 range [+0.8213,+0.9127]; s2 mean -0.0031 std 0.0693 range [-0.1477,+0.0875]; s3 mean -0.1193 std 0.1878 range [-0.3534,+0.1300]
Train Nov3 cluster+ n=5; gamma mean +0.3250 std 0.0376 range [+0.2902,+0.3760]; theta_K mean -0.4017 std 0.0249 range [-0.4224,-0.3643]; theta_M mean -0.0048 std 0.0140 range [-0.0207,+0.0105]; s1 mean +0.8760 std 0.0263 range [+0.8460,+0.9127]; s2 mean -0.0147 std 0.0747 range [-0.1477,+0.0298]; s3 mean -0.1561 std 0.2111 range [-0.3534,+0.1300]
Train Nov3 cluster- n=3; gamma mean -0.2863 std 0.0521 range [-0.3398,-0.2358]; theta_K mean -0.3412 std 0.0387 range [-0.3762,-0.2996]; theta_M mean +0.0453 std 0.0745 range [-0.0198,+0.1265]; s1 mean -0.8592 std 0.0329 range [-0.8798,-0.8213]; s2 mean +0.0163 std 0.0690 range [-0.0502,+0.0875]; s3 mean -0.0579 std 0.1591 range [-0.1799,+0.1221]
Train QAOA unsigned n=8; gamma mean -0.0672 std 0.2981 range [-0.3378,+0.3613]; beta mean +0.1042 std 0.4029 range [-0.3935,+0.4049]
Train QAOA cluster+ n=5; gamma mean -0.2804 std 0.0418 range [-0.3378,-0.2214]; beta mean +0.3961 std 0.0053 range [+0.3917,+0.4049]
Train QAOA cluster- n=3; gamma mean +0.2882 std 0.0661 range [+0.2328,+0.3613]; beta mean -0.3823 std 0.0097 range [-0.3935,-0.3762]

B* wall 34.59891629219055. Train fit wall 1.5551981925964355. Test wall 1.4860279560089111. Sum cell wall 7.1s. Peak RSS KiB 181872.

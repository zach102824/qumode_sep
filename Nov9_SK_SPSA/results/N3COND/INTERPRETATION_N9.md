# N3COND N9 reading (development only)

Parent pid 7281. Clock time 11.9 s for 192 fits. Summed worker time 47.5 s. Failures 0. Every fit spent its full Q = qL queries (128, 384, 512, 1024, 1536, or 3072). Comparison queries 212992. Axial scale queries 832, all on seeds 10010–10013. The unused simultaneous-probe file is 512 queries and is not an input to C1. One post-hoc state rebuild per fit is not charged. GS preparation is classical and not charged. Fidelity is post hoc.

**Do not adopt C1.** The preregistered holdout cell is N9, L3, q=1024. Paired C1−C0 on Nov3: median ΔF −0.012, mean ΔF −0.020 with 95% CI [−0.068, +0.019], median ΔE +0.023, mean ΔE +0.043 with CI [+0.002, +0.103]. The energy CI is entirely on the side of higher energy for C1. The adoption rule (median ΔF ≥ 0.02 and median ΔE < 0) fails. Dev at the same cell leans the other way (median ΔE −0.030) and is not the predictive check.

The loss is larger where the budget is small. Holdout Nov3 L1 q=1024: median ΔF −0.024, 0/4 higher F, median ΔE +0.045, 4/4 higher energy. Holdout L1 q=128: median ΔF −0.106. QAOA's own C1−C0 at L1 q=1024 is about −0.003 in median F. Conditioning did not help QAOA either.

Scalar Nov3 still beats scalar QAOA at L1 on these seeds. Holdout q=1024: median F 0.693 vs 0.575, median ΔF +0.122, 4/4, median ΔE −0.202. At L3 q=1024 the fidelity median still favors Nov3 (0.890 vs 0.812, ΔF +0.034, 4/4) but the energy median does not (ΔE +0.015, mean −0.005, CI crosses 0). Tails at holdout L3 q=1024: Nov3 C0 min F 0.764, all four F≥0.5. This is a different seed band from GR and a different ledger from the local continuation recipe. It is not a change to GR's benchmark.

The frozen scale is why C1 can lose. At N9 L1 the axial |σ ∂E| medians are gamma 0.42, thetaK 0.14, thetaM 0.049, s1 0.20, s2 0.025, s3 0.0016. s3 against gamma is far outside the clip [0.25, 4], so two coordinates sit on the clip (four at L3). The frozen gain is then about 38, while C0 recomputes the gain at every start. C1 still spends the 8 calibration queries and discards them.

N12 is the precommitted next batch, with a new scale freeze on the same dev seeds and no change to the 0.02 rule. N15 and N18 stay off: the gate needs both N9 and N12 holdout median ΔF ≥ 0.02 at L3 q=1024, and N9 is already below that.

arXiv:2204.02923 is not a recipe here. Its entangler is a free M_nm under natural gradient. Natural gradient is outside this stream, and that M_nm is not the cavity-block feature.

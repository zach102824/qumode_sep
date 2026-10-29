"""Promotion gate: S_comb + n9 med, never ratio alone."""
from qtimes_cost.lib.promote import (
    BOX_NEW,
    BOX_R014,
    CLOUD_L12,
    beats_box_reval,
    fork_promote,
    promotion_decision,
    ratio_9_6,
)


def test_r014_reference_ratio():
    assert abs(ratio_9_6(BOX_R014["med6"], BOX_R014["med9"]) - BOX_R014["ratio_9_6"]) < 1e-9


def test_first_candidate_kept():
    ok, why = promotion_decision({"s_comb": 4.0, "med6": 30.0, "med9": 40.0}, None)
    assert ok and why == "first"


def test_worse_score_rejected():
    champ = {"s_comb": 3.2, "med6": 30.0, "med9": 40.0}
    ok, why = promotion_decision({"s_comb": 3.5, "med6": 20.0, "med9": 30.0}, champ)
    assert not ok and why == "s_comb"


def test_ratio_improvement_with_worse_n9_rejected():
    # Healthier ratio (1.125 vs 1.75) but n9 got worse and S only looks better
    # because n6 collapsed upward... here S is better AND ratio is better,
    # n9 is worse: junk on ratio alone.
    champ = {"s_comb": 3.4, "med6": 40.0, "med9": 70.0}
    cand = {"s_comb": 3.1, "med6": 80.0, "med9": 90.0}
    assert ratio_9_6(80, 90) < ratio_9_6(40, 70)
    ok, why = promotion_decision(cand, champ)
    assert not ok and why == "ratio_only"


def test_s_win_that_regresses_n9_rejected():
    champ = {"s_comb": 3.5, "med6": 40.0, "med9": 50.0}
    # n6 much better, n9 worse, ratio also worse so not the ratio_only branch
    cand = {"s_comb": 3.0, "med6": 15.0, "med9": 70.0}
    ok, why = promotion_decision(cand, champ)
    assert not ok and why == "n9_regressed"


def test_real_n9_improvement_promotes():
    champ = {"s_comb": 3.5, "med6": 30.0, "med9": 45.0, "leak9": 0.08}
    cand = {"s_comb": 3.2, "med6": 28.0, "med9": 36.0, "leak9": 0.07}
    ok, why = promotion_decision(cand, champ)
    assert ok and why == "promote"


def test_unhealthy_ratio_veto_unless_n9_clearly_better():
    champ = {"s_comb": 3.6, "med6": 30.0, "med9": 40.0}
    fat = {"s_comb": 3.2, "med6": 20.0, "med9": 70.0}  # ratio 3.5, n9 worse
    ok, why = promotion_decision(fat, champ)
    assert not ok
    assert why in ("n9_regressed", "unhealthy_ratio", "ratio_only")

    # ratio still high, but n9 improved by more than 10% → allow
    clear = {"s_comb": 3.2, "med6": 12.0, "med9": 32.0}  # ratio 2.67, n9 32 vs 40
    ok, why = promotion_decision(clear, champ)
    assert ok and why == "promote"


def test_beats_box_reval_requires_both_and_healthy_ratio():
    # Box reval40 bar is 14.5 / 19.5. Small-val 10.2/18 is not the claim bar.
    assert BOX_NEW["reval40_med6"] == 14.5
    assert BOX_NEW["reval40_med9"] == 19.5
    win = beats_box_reval(14.0, 18.0)
    assert win["full_win"] and win["beats_n6"] and win["beats_n9"]
    # n9 better, n6 worse: not a full win
    partial = beats_box_reval(20.0, 17.0)
    assert partial["beats_n9"] and not partial["beats_n6"] and not partial["full_win"]
    # both medians beat the bar but ratio 18/8 = 2.25 is outside [0.75, 2]
    unhealthy = beats_box_reval(8.0, 18.0)
    assert unhealthy["beats_n6"] and unhealthy["beats_n9"] and not unhealthy["ratio_healthy"]
    assert not unhealthy["full_win"]
    miss = beats_box_reval(14.5, 19.5)  # not strictly better
    assert not miss["full_win"]


def test_fork_promote_needs_both_val_medians_and_intact_test():
    champ = {"med6": CLOUD_L12["val40_med6"], "med9": CLOUD_L12["val40_med9"], "s_comb": CLOUD_L12["val40_S"]}
    caps = {"med6": CLOUD_L12["test_med6"] * 1.20, "med9": CLOUD_L12["test_med9"] * 1.20}
    good_val = {"med6": 11.0, "med9": 15.5, "s_comb": 3.1}
    good_test = {"med6": 15.0, "med9": 16.5}
    ok, why = fork_promote(good_val, good_test, champ, test_caps=caps)
    assert ok and why == "promote"
    # n9 not improved
    ok, why = fork_promote({"med6": 11.0, "med9": 17.5, "s_comb": 3.1}, good_test, champ, test_caps=caps)
    assert not ok and why == "val_med9"
    # val wins but test n9 jumps well past the cap
    ok, why = fork_promote(good_val, {"med6": 15.0, "med9": 30.0}, champ, test_caps=caps)
    assert not ok and why == "test_wreck"
    # ratio outside the band
    ok, why = fork_promote({"med6": 6.0, "med9": 15.0, "s_comb": 3.0}, good_test, champ, test_caps=caps)
    assert not ok and why == "ratio"


def test_warm_start_repeats_last_layer():
    import torch
    from qtimes_cost.lib.cloud_worker import warm_start_gnn
    from qtimes_cost.lib.gnn import FactorGNN, GNNArch

    arch = GNNArch(H=8, rounds=1, layernorm=True, residual=True, degree_features=True)
    src = FactorGNN(arch, n=6, m=8, L=2, seed=0)
    with torch.no_grad():
        src.head_q.bias.copy_(torch.tensor([0.25, -0.5]))
    dst = FactorGNN(arch, n=6, m=8, L=4, seed=3)
    warm_start_gnn(dst, src.state_dict(), src_L=2)
    q = dst.head_q.bias.detach()
    assert torch.allclose(q[:2], src.head_q.bias.detach())
    assert torch.allclose(q[2:], src.head_q.bias.detach()[-1].expand(2))
    # same depth is an exact copy of the parent heads
    same = FactorGNN(arch, n=6, m=8, L=2, seed=9)
    warm_start_gnn(same, src.state_dict(), src_L=2)
    assert torch.allclose(same.head_alpha0.weight, src.head_alpha0.weight)
    assert torch.allclose(same.head_A.weight, src.head_A.weight)


def test_copy_shared_body_skips_cavity_heads():
    import torch
    from qtimes_cost.lib.cloud_worker import copy_shared_body
    from qtimes_cost.lib.gnn import FactorGNN, GNNArch

    arch = GNNArch(H=8, rounds=1, layernorm=True, residual=True, degree_features=True)
    src = FactorGNN(arch, n=9, m=12, L=2, seed=0)
    dst = FactorGNN(arch, n=12, m=16, L=2, seed=1)
    fresh = dst.head_alpha0.weight.detach().clone()
    copied = copy_shared_body(dst, src.state_dict())
    assert "head_q.weight" in copied
    assert not any(k.startswith("head_alpha0") or k.startswith("head_A") or k.startswith("head_TH") for k in copied)
    assert torch.allclose(dst.head_q.weight, src.head_q.weight)
    assert torch.allclose(dst.head_alpha0.weight, fresh)


def test_transfer_gnn_repeats_last_cavity():
    import torch
    from qtimes_cost.lib.cloud_worker import transfer_gnn_n
    from qtimes_cost.lib.gnn import FactorGNN, GNNArch

    arch = GNNArch(H=8, rounds=1, layernorm=True, residual=True, degree_features=True)
    src = FactorGNN(arch, n=9, m=12, L=2, seed=0)
    with torch.no_grad():
        src.head_alpha0.bias.copy_(torch.arange(6, dtype=torch.float32))
    dst = FactorGNN(arch, n=12, m=16, L=2, seed=1)
    src_C = transfer_gnn_n(dst, src.state_dict())
    assert src_C == 3
    got = dst.head_alpha0.bias.detach().reshape(4, 2)
    parent = src.head_alpha0.bias.detach().reshape(3, 2)
    assert torch.allclose(got[:3], parent)
    assert torch.allclose(got[3], parent[2])
    assert torch.allclose(dst.head_q.weight, src.head_q.weight)


def test_leak_veto():
    champ = {"s_comb": 3.5, "med6": 30.0, "med9": 40.0, "leak9": 0.05}
    cand = {"s_comb": 3.2, "med6": 29.0, "med9": 39.0, "leak9": 0.50}
    ok, why = promotion_decision(cand, champ)
    assert not ok and why == "leak"

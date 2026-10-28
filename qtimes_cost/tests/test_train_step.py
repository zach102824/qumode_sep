"""One short train step on CPU; gradients finite."""
from __future__ import annotations
import numpy as np
import torch
import pytest
from pathlib import Path

from qtimes_cost.lib.gnn import FactorGNN, GNNArch
from qtimes_cost.lib.ansatz import AnsatzConfig, MultiCavityAnsatz
from qtimes_cost.lib.sat_gen import clause_count, gen_instance
from qtimes_cost.lib.train_loop import _instance_loss, should_promote, save_gnn_best, load_gnn_best, combined_score
from qtimes_cost.lib.device import resolve_device, dual_devices, cuda_count


def test_one_train_step_cpu_grads_finite(tmp_path):
    n, L, nf = 6, 2, 16
    m = clause_count(n)
    arch = GNNArch(H=16, rounds=1, lr=1e-3, wd=0.0)
    device = torch.device("cpu")
    gnn = FactorGNN(arch, n=n, m=m, L=L, seed=0).to(device)
    cfg = AnsatzConfig(n=n, L=L, nf=nf)
    sim = MultiCavityAnsatz(cfg, device=device).to(device)
    # fresh satisfiable instance
    _, cl, E = gen_instance(n, 51_000_000_000 + 10_000_000 * n + 7)
    cls_v = torch.as_tensor(np.array([v for v, s in cl], dtype=np.int64), device=device)
    cls_s = torch.as_tensor(np.array([s for v, s in cl], dtype=np.float32), device=device)
    Eb = torch.as_tensor(E, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    opt.zero_grad()
    heads = gnn.forward_graph(cls_v, cls_s)
    out = sim.forward(Eb, heads["alpha0"], heads["A"], heads["TH"], heads["q_logits"], gamma=heads["gamma"])
    loss = _instance_loss(out)
    loss.backward()
    for p in gnn.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "non-finite grad"
    opt.step()
    assert torch.isfinite(loss).item()


def test_best_only_promote(tmp_path):
    assert should_promote(1.0, 2.0) is True
    assert should_promote(2.0, 1.0) is False
    assert should_promote(1.0, 1.0) is False
    assert should_promote(0.9, 1.0, min_improve=0.05) is True
    assert should_promote(0.99, 1.0, min_improve=0.05) is False

    # worse score must not overwrite; better must
    from qtimes_cost.lib.gnn import BASELINE_ARCH
    runs = tmp_path / "runs"
    arch = GNNArch(H=8, rounds=1)
    good_metrics = {
        6: {"S": 1.0, "median_inv_p": 2.0, "B90_inv_p": 3.0},
        9: {"S": 1.0, "median_inv_p": 2.0, "B90_inv_p": 3.0},
    }
    bad_metrics = {
        6: {"S": 5.0, "median_inv_p": 20.0, "B90_inv_p": 30.0},
        9: {"S": 5.0, "median_inv_p": 20.0, "B90_inv_p": 30.0},
    }
    # fake tiny state
    state = {"dummy": torch.tensor([1.0])}
    # monkey: save uses state_dict keys — just store tensors
    save_gnn_best(arch, {6: state}, good_metrics, runs_dir=runs)
    best_s = combined_score(good_metrics)
    cand_s = combined_score(bad_metrics)
    assert not should_promote(cand_s, best_s)
    # load back
    a2, p2 = load_gnn_best(runs_dir=runs)
    assert a2 is not None
    assert a2.H == 8
    assert 6 in p2

    # better promotes
    better = {
        6: {"S": 0.5, "median_inv_p": 1.5, "B90_inv_p": 2.0},
        9: {"S": 0.5, "median_inv_p": 1.5, "B90_inv_p": 2.0},
    }
    assert should_promote(combined_score(better), best_s)
    save_gnn_best(arch, {6: {"dummy": torch.tensor([2.0])}}, better, runs_dir=runs)
    a3, p3 = load_gnn_best(runs_dir=runs)
    assert float(p3[6]["dummy"][0]) == 2.0


def test_dual_device_path():
    """If 2+ CUDA devices, run two tiny forwards on cuda:0 and cuda:1; else skip."""
    if cuda_count() < 2:
        pytest.skip(f"need ≥2 CUDA devices for dual-GPU test (have {cuda_count()})")
    n, L, nf = 6, 1, 16
    m = clause_count(n)
    arch = GNNArch(H=8, rounds=1)
    _, cl, E = gen_instance(n, 51_000_000_000 + 10_000_000 * n + 2)
    cls_v = np.array([v for v, s in cl], dtype=np.int64)
    cls_s = np.array([s for v, s in cl], dtype=np.float32)

    def run_on(dev):
        gnn = FactorGNN(arch, n=n, m=m, L=L, seed=0).to(dev)
        sim = MultiCavityAnsatz(AnsatzConfig(n=n, L=L, nf=nf), device=dev).to(dev)
        heads = gnn.forward_graph(torch.as_tensor(cls_v, device=dev), torch.as_tensor(cls_s, device=dev))
        out = sim.forward(
            torch.as_tensor(E, dtype=torch.float32, device=dev),
            heads["alpha0"], heads["A"], heads["TH"], heads["q_logits"], gamma=heads["gamma"],
        )
        return float(out["inv_p"].cpu())

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:
        f0 = ex.submit(run_on, torch.device("cuda:0"))
        f1 = ex.submit(run_on, torch.device("cuda:1"))
        a, b = f0.result(), f1.result()
    assert a > 0 and b > 0


def test_resolve_device_cpu_fallback():
    d = resolve_device("cpu")
    assert d.type == "cpu"
    pair = dual_devices()
    assert len(pair) == 2

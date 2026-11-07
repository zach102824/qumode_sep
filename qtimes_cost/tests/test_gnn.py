"""GNN forward shapes for heads."""
from __future__ import annotations
import numpy as np
import torch
import pytest

from qtimes_cost.lib.gnn import FactorGNN, GNNArch, BASELINE_ARCH, SizeGeneralGNN
from qtimes_cost.lib.packing import NU, n_cavities
from qtimes_cost.lib.sat_gen import clause_count, gen_instance


@pytest.mark.parametrize("n,L", [(6, 5), (9, 3)])
def test_gnn_head_shapes(n, L):
    m = clause_count(n)
    arch = GNNArch(H=32, rounds=2, layernorm=True, residual=True)
    gnn = FactorGNN(arch, n=n, m=m, L=L, seed=0)
    _, cl, _ = gen_instance(n, 51_000_000_000 + 10_000_000 * n + 1, compute_E=False)
    cls_v = np.array([v for v, s in cl], dtype=np.int64)
    cls_s = np.array([s for v, s in cl], dtype=np.float32)
    heads = gnn.forward_graph(torch.as_tensor(cls_v), torch.as_tensor(cls_s))
    C = n_cavities(n)
    assert heads["alpha0"].shape == (C, 2)
    assert heads["A"].shape == (C, L, 2)
    assert heads["TH"].shape == (C, L, NU)
    assert heads["q_logits"].shape == (L,)
    assert heads["gamma"].ndim == 0


def test_size_general_one_module_two_widths():
    """Same parameters serve n=6 (C=2) and n=12 (C=4). No per-n Linear(H, C*2)."""
    arch = GNNArch(H=16, rounds=1, layernorm=True, residual=True, degree_features=True)
    gnn = SizeGeneralGNN(arch, L=3, seed=0)
    names = [name for name, _ in gnn.named_parameters()]
    assert not any("head_alpha0" in name or "head_A" in name or "head_TH" in name for name in names)
    for n in (6, 12):
        _, cl, _ = gen_instance(n, 51_000_000_000 + 10_000_000 * n + 3, compute_E=False)
        cls_v = torch.as_tensor(np.array([v for v, s in cl], dtype=np.int64))
        cls_s = torch.as_tensor(np.array([s for v, s in cl], dtype=np.float32))
        heads = gnn.forward_graph(cls_v, cls_s, n=n)
        C = n_cavities(n)
        assert heads["alpha0"].shape == (C, 2)
        assert heads["A"].shape == (C, 3, 2)
        assert heads["TH"].shape == (C, 3, NU)
        assert heads["q_logits"].shape == (3,)
        assert heads["gamma"].ndim == 0


def test_n9_snap_residual_is_zero_at_init_and_inactive_at_other_n():
    arch = GNNArch(H=16, rounds=1, layernorm=True, residual=True, degree_features=True)
    gnn = SizeGeneralGNN(arch, L=2, seed=1)
    _, cl, _ = gen_instance(9, 51_000_000_000 + 10_000_000 * 9 + 4, compute_E=False)
    cv = torch.as_tensor(np.array([v for v, s in cl], dtype=np.int64))
    cs = torch.as_tensor(np.array([s for v, s in cl], dtype=np.float32))
    before = gnn.forward_graph(cv, cs, n=9)["TH"].detach().clone()
    gnn.enable_n9_snap()
    after = gnn.forward_graph(cv, cs, n=9)["TH"].detach()
    assert torch.allclose(before, after)
    _, cl6, _ = gen_instance(6, 51_000_000_000 + 10_000_000 * 6 + 4, compute_E=False)
    cv6 = torch.as_tensor(np.array([v for v, s in cl6], dtype=np.int64))
    cs6 = torch.as_tensor(np.array([s for v, s in cl6], dtype=np.float32))
    th6 = gnn.forward_graph(cv6, cs6, n=6)["TH"]
    assert th6.shape == (2, 2, NU)

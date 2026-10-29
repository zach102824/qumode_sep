"""GNN forward shapes for shared-mixer heads (independent of C)."""
from __future__ import annotations
import numpy as np
import torch
import pytest

from explore_OCT29.lib.gnn import FactorGNN, GNNArch, BASELINE_ARCH
from explore_OCT29.lib.packing import NU, n_cavities
from explore_OCT29.lib.sat_gen import clause_count, gen_instance


@pytest.mark.parametrize("n,L", [(6, 5), (9, 3), (12, 5)])
def test_gnn_shared_head_shapes(n, L):
    m = clause_count(n)
    arch = GNNArch(H=32, rounds=2, layernorm=True, residual=True)
    gnn = FactorGNN(arch, n=n, m=m, L=L, seed=0)
    _, cl, _ = gen_instance(n, 51_000_000_000 + 10_000_000 * n + 1, compute_E=False)
    cls_v = np.array([v for v, s in cl], dtype=np.int64)
    cls_s = np.array([s for v, s in cl], dtype=np.float32)
    heads = gnn.forward_graph(torch.as_tensor(cls_v), torch.as_tensor(cls_s))
    assert heads["alpha0"].shape == (2,)
    assert heads["A"].shape == (L, 2)
    assert heads["TH"].shape == (L, NU)
    assert heads["gamma"].shape == (L,)
    assert "q_logits" not in heads
    # no per-cavity mixer dims
    C = n_cavities(n)
    assert heads["A"].shape[0] == L and heads["A"].ndim == 2
    assert arch.mixer_params_per_layer() == 11
    # weight shapes identical across n (transfer)
    n_params = sum(p.numel() for p in gnn.parameters())
    gnn2 = FactorGNN(arch, n=6 if n != 6 else 9, m=clause_count(6 if n != 6 else 9), L=L, seed=0)
    assert sum(p.numel() for p in gnn2.parameters()) == n_params


def test_zero_shot_state_load_n9_to_n12():
    """Shared heads: n=9 checkpoint loads into n=12 FactorGNN."""
    L = 3
    arch = GNNArch(H=16, rounds=1)
    g9 = FactorGNN(arch, n=9, m=clause_count(9), L=L, seed=0)
    g12 = FactorGNN(arch, n=12, m=clause_count(12), L=L, seed=1)
    missing, unexpected = g12.load_state_dict(g9.state_dict(), strict=True)
    assert missing == [] and unexpected == []

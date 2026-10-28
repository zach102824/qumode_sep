"""GNN forward shapes for heads."""
from __future__ import annotations
import numpy as np
import torch
import pytest

from qtimes_cost.lib.gnn import FactorGNN, GNNArch, BASELINE_ARCH
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

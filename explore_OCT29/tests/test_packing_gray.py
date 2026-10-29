"""Packing / Gray grid consistency (n=6,9)."""
from __future__ import annotations
import numpy as np
import pytest

from explore_OCT29.lib.packing import K_BITS, NU, n_cavities, state_dim
from explore_OCT29.lib.gray_grid import build_grid_to_x, gray, ungray


@pytest.mark.parametrize("n", [6, 9])
def test_packing_basic(n):
    assert n % K_BITS == 0
    C = n_cavities(n)
    assert C == n // 3
    assert NU == 8
    assert state_dim(n) == 2 ** n
    assert state_dim(n) == NU ** C


def test_gray_roundtrip():
    for i in range(8):
        assert ungray(gray(i)) == i


@pytest.mark.parametrize("n", [6, 9])
def test_grid_to_x_bijection(n):
    g = build_grid_to_x(n)
    assert g.shape == (NU,) * n_cavities(n)
    flat = g.reshape(-1)
    assert flat.min() == 0
    assert flat.max() == 2 ** n - 1
    assert len(np.unique(flat)) == 2 ** n

"""Fixed K=3 Gray bits per cavity packing. C = n/3, NU = 8, state dim = 8^C = 2^n."""
from __future__ import annotations

K_BITS = 3
NU = 1 << K_BITS  # 8


def n_cavities(n: int) -> int:
    if n % K_BITS != 0:
        raise ValueError(f"n={n} must be a multiple of {K_BITS}")
    return n // K_BITS


def state_dim(n: int) -> int:
    return NU ** n_cavities(n)  # == 2**n


def ladder_ns():
    return (6, 9, 12, 15, 18, 21)


def train_ns():
    return (6, 9)


def eval_ns():
    return (12, 15, 18, 21)


def default_nf(n: int) -> int:
    """Simulation Fock truncation per cavity (computational NU + pad)."""
    # modest pad; leakage check uses a larger NF_CHK
    return max(24, NU + 16)


def default_nf_chk(n: int) -> int:
    return max(40, NU + 32)

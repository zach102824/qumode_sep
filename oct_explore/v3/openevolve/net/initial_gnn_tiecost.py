"""EVOLVE adaptive GNN training rule for tied-cost M-C^L (SNAP+D -> cost; cost unitary tied across layers).
rule(L) -> hypers dict. Always include train_ns with 6 and 8; may add 9 and/or 12.
Allowed: numpy, math. No file/os/sys/network.
"""
import math

# EVOLVE-BLOCK-START
# FAMILY: baseline — Part1 defaults, L-aware steps/lr mild schedule

def rule(L: int) -> dict:
    """Adaptive training rule conditioned on depth L."""
    L = int(L)
    # capacity: slightly more H/rounds at larger L
    if L <= 5:
        H, rounds, steps, lr = 128, 4, 1000, 1.2e-3
    elif L <= 10:
        H, rounds, steps, lr = 160, 5, 1200, 1.0e-3
    else:
        H, rounds, steps, lr = 192, 5, 1500, 8e-4
    return dict(
        arch="mp",
        H=H,
        rounds=rounds,
        lr=lr,
        steps=steps,
        bs=250,
        wd=1e-4,
        leakpen=0.1,
        tailw=1.0,
        ntrain=10000,
        train_ns=(6, 8),  # OE may add 9 and/or 12
        amax=1.8,
        seed=0,
    )
# EVOLVE-BLOCK-END

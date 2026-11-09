"""Energy-legal post-selection. No Hamiltonian and no ground state."""
from __future__ import annotations


def energy_best(rows):
    if not rows:
        raise ValueError("no rows")
    return min(rows, key=lambda row: (float(row["E"]), int(row["r"]), str(row["label"])))


def legal_pool(rows, e_qaoa, tol):
    limit = float(e_qaoa) + float(tol)
    return [row for row in rows if float(row["E"]) <= limit]


def pick(rows, e_qaoa, tol):
    """Highest F_sub among energy-legal rows. None when the pool is empty.

    Ties: lower energy, then lower repeat, then label.
    """
    pool = legal_pool(rows, e_qaoa, tol)
    if not pool:
        return None
    return min(pool, key=lambda row: (-float(row["F"]), float(row["E"]), int(row["r"]), str(row["label"])))


def unrestricted(rows):
    """Highest F_sub in the whole list. Descriptive only."""
    return min(rows, key=lambda row: (-float(row["F"]), float(row["E"]), int(row["r"]), str(row["label"])))

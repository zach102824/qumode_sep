"""J-only cavity and bit-role assignments.

No function here accepts a ground state. Random arms read no J entries.
anti-R reverses the R-ordered triple. Gray is a bit labeling, not a new grouping.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hamiltonian import block_coeff, hat, pair_diag, upper


@dataclass
class Packing:
    name: str
    groups: list
    encoding: str
    ops: dict = field(default_factory=dict)

    @property
    def partition(self) -> tuple:
        return tuple(sorted(tuple(sorted(g)) for g in self.groups))


def interleaved_groups(n: int) -> list:
    m = n // 3
    return [[a, a + m, a + 2 * m] for a in range(m)]


def _row_norms(J: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(J * J, axis=1))


def greedy_groups(J: np.ndarray) -> tuple[list, dict]:
    n = J.shape[0]
    remaining = list(range(n))
    groups = []
    ops = {"triple_evals": 0, "pair_reads": 0, "gs_reads": 0, "rule": "greedy_sequential"}
    while remaining:
        best = None
        best_score = None
        for a in range(len(remaining)):
            for b in range(a + 1, len(remaining)):
                for c in range(b + 1, len(remaining)):
                    i, j, k = remaining[a], remaining[b], remaining[c]
                    score = abs(float(J[i, j])) + abs(float(J[i, k])) + abs(float(J[j, k]))
                    ops["triple_evals"] += 1
                    ops["pair_reads"] += 3
                    triple = (i, j, k)
                    if best is None or score > best_score or (score == best_score and triple < best):
                        best = triple
                        best_score = score
        assert best is not None
        groups.append([best[0], best[1], best[2]])
        for s in best:
            remaining.remove(s)
    return groups, ops


def apply_R(group: list, J: np.ndarray, norms: np.ndarray, ops: dict) -> list:
    i, j, k = group
    pairs = [(i, j), (i, k), (j, k)]

    def key(p):
        a, b = p
        return (-abs(float(J[a, b])), min(a, b), max(a, b))

    ops["pair_reads"] = ops.get("pair_reads", 0) + 3
    a, b = min(pairs, key=key)
    third = i + j + k - a - b
    bit0, bit1 = (a, b) if (norms[a], a) <= (norms[b], b) else (b, a)
    return [int(bit0), int(bit1), int(third)]


def random_groups(seed: int, n: int, r: int) -> tuple[list, dict]:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(n), 1000 + int(r), 2]))
    perm = rng.permutation(n)
    groups = [[int(perm[3 * c + t]) for t in range(3)] for c in range(n // 3)]
    ops = {"triple_evals": 0, "pair_reads": 0, "permutation_draws": int(n), "gs_reads": 0, "rule": f"random_{r}"}
    return groups, ops


def build_named_groups(J: np.ndarray, seed: int) -> dict:
    """Return ordered triples for every GR rule. Identity G groups are sorted."""
    n = J.shape[0]
    norms = _row_norms(J)
    g_groups, g_ops = greedy_groups(J)
    g_ops["row_norm_muls"] = int(n * n)
    g_ops["gs_reads"] = 0
    out = {}
    out["inter_id"] = (interleaved_groups(n), {"triple_evals": 0, "pair_reads": 0, "gs_reads": 0, "rule": "interleaved"})
    out["G_id"] = (g_groups, dict(g_ops))
    r_ops = dict(g_ops)
    r_groups = [apply_R(g, J, norms, r_ops) for g in g_groups]
    r_ops["rule"] = "G+R"
    out["G_R"] = (r_groups, r_ops)
    anti_ops = dict(r_ops)
    anti_ops["rule"] = "G+antiR"
    out["G_antiR"] = ([[g[2], g[1], g[0]] for g in r_groups], anti_ops)
    inter_r_ops = {"triple_evals": 0, "pair_reads": 0, "row_norm_muls": int(n * n), "gs_reads": 0, "rule": "inter+R"}
    out["inter_R"] = ([apply_R(g, J, norms, inter_r_ops) for g in interleaved_groups(n)], inter_r_ops)
    for r in range(3):
        groups, ops = random_groups(seed, n, r)
        out[f"rand{r}"] = (groups, ops)
    for name, (groups, ops) in out.items():
        if ops.get("gs_reads", 0) != 0:
            raise RuntimeError(f"{name} read a ground state")
    return out


def packing_from(name: str, groups: list, encoding: str, ops: dict) -> Packing:
    return Packing(name=name, groups=[[int(x) for x in g] for g in groups], encoding=encoding, ops=dict(ops))


def feature_M(J: np.ndarray, groups: list, rms_j: float, z: np.ndarray):
    raw = block_coeff(J, groups)
    M_hat, scale = hat(raw, rms_j)
    diag = pair_diag(M_hat, z)
    return raw, M_hat, float(scale), diag


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    na = np.linalg.norm(aa)
    nb = np.linalg.norm(bb)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(aa, bb) / (na * nb))


def m_correlation(J, groups, rms_j, z, hc) -> float:
    _raw, _hat, _scale, diag = feature_M(J, groups, rms_j, z)
    return pearson(diag, hc)

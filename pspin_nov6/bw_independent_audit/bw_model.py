"""Staggered Baxter-Wu torus, built without any code from the other branch.

Site (x, y) has index x + Lx * y. Up faces are
{(x,y), (x+1,y), (x+1,y+1)}. Down faces are
{(x,y), (x,y+1), (x+1,y+1)}. Indices are periodic.

H = -sum_up ZZZ + lam * sum_down ZZZ - Gamma * sum X.

Bit i of a basis integer is qubit i. Bit 0 is |0> with Z=+1.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np


def site(x: int, y: int, lx: int, ly: int) -> int:
    return (x % lx) + lx * (y % ly)


def faces(lx: int, ly: int) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    up, down = [], []
    for y in range(ly):
        for x in range(lx):
            up.append(tuple(sorted((
                site(x, y, lx, ly),
                site(x + 1, y, lx, ly),
                site(x + 1, y + 1, lx, ly),
            ))))
            down.append(tuple(sorted((
                site(x, y, lx, ly),
                site(x, y + 1, lx, ly),
                site(x + 1, y + 1, lx, ly),
            ))))
    return up, down


def clauses(lx: int, ly: int, lam: float = 1.0) -> list[tuple[str, tuple[int, int, int], float]]:
    up, down = faces(lx, ly)
    out = [("up", f, -1.0) for f in up] + [("down", f, float(lam)) for f in down]
    return out


def term_hash(lx: int, ly: int, lam: float = 1.0) -> str:
    lines = []
    for kind, qubits, coeff in clauses(lx, ly, lam):
        lines.append(f"{kind} {coeff:.1f} {qubits[0]} {qubits[1]} {qubits[2]}")
    blob = "\n".join(sorted(lines)).encode()
    return hashlib.sha256(blob).hexdigest()


def cost_diagonal(lx: int, ly: int, lam: float = 1.0) -> np.ndarray:
    """Vectorized parity diagonal. Independent of any Kronecker builder."""
    n = lx * ly
    xs = np.arange(1 << n, dtype=np.uint32)
    acc = np.zeros(xs.shape[0], dtype=np.float64)
    for _kind, qubits, coeff in clauses(lx, ly, lam):
        parity = np.zeros(xs.shape[0], dtype=np.uint32)
        for q in qubits:
            parity ^= (xs >> np.uint32(q)) & np.uint32(1)
        prod = 1.0 - 2.0 * parity.astype(np.float64)
        acc += coeff * prod
    return acc


def apply_sum_x(psi: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(psi)
    for i in range(n):
        v = np.asarray(psi).reshape(1 << (n - i - 1), 2, 1 << i)
        o = out.reshape(v.shape)
        o[:, 0, :] += v[:, 1, :]
        o[:, 1, :] += v[:, 0, :]
        psi = v.reshape(-1)
        out = o.reshape(-1)
    return out


def apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """exp(-i beta X) on every qubit. Qubit 0 is the fastest index."""
    cb = math.cos(float(beta))
    sb = math.sin(float(beta))
    out = np.array(psi, dtype=np.complex128, copy=True)
    for i in range(n):
        v = out.reshape(1 << (n - i - 1), 2, 1 << i)
        a = np.array(v[:, 0, :], copy=True)
        b = np.array(v[:, 1, :], copy=True)
        v[:, 0, :] = cb * a - 1j * sb * b
        v[:, 1, :] = cb * b - 1j * sb * a
        out = v.reshape(-1)
    return out


def evolve(diag: np.ndarray, gammas, betas, n: int, order: str = "cost_then_mixer") -> np.ndarray:
    """Layer list is applied from index 0 (closest to |+>)."""
    psi = np.full(1 << n, 2.0 ** (-0.5 * n), dtype=np.complex128)
    for g, b in zip(np.atleast_1d(np.asarray(gammas, dtype=np.float64)),
                    np.atleast_1d(np.asarray(betas, dtype=np.float64))):
        if order == "cost_then_mixer":
            psi = psi * np.exp(-1j * float(g) * diag)
            psi = apply_mixer(psi, float(b), n)
        elif order == "mixer_then_cost":
            psi = apply_mixer(psi, float(b), n)
            psi = psi * np.exp(-1j * float(g) * diag)
        else:
            raise ValueError(order)
    return psi


def translate_index(x: int, lx: int, ly: int, dx: int, dy: int) -> int:
    n = lx * ly
    y = 0
    for i in range(n):
        if (x >> i) & 1:
            ix = i % lx
            iy = i // lx
            j = site(ix + dx, iy + dy, lx, ly)
            y |= 1 << j
    return y


def shift_state(psi: np.ndarray, lx: int, ly: int, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(psi)
    for x in range(psi.shape[0]):
        out[translate_index(int(x), lx, ly, dx, dy)] += psi[x]
    return out


def coloring_conflicts(lx: int, ly: int) -> dict:
    """(x+y) mod 3 on every triangle edge, plus forced-propagation search.

    One triangle has 3! proper colorings. Each neighboring triangle shares
    an edge, so it is forced. If the face-adjacency graph is connected, the
    whole torus is forced by the first triangle, and a conflict is a proof
    that no 3-face-coloring exists.
    """
    up, down = faces(lx, ly)
    all_faces = up + down
    # edges of the formula
    mono = []
    coord = {}
    for y in range(ly):
        for x in range(lx):
            coord[site(x, y, lx, ly)] = (x + y) % 3
    seen_edges = set()
    for f in all_faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[0], f[2])):
            e = tuple(sorted((a, b)))
            if e in seen_edges:
                continue
            seen_edges.add(e)
            if coord[a] == coord[b]:
                mono.append(e)

    def propagate(seed: dict) -> str:
        colors = dict(seed)
        pending = True
        guard = 0
        while pending and guard < (lx * ly + 5):
            pending = False
            guard += 1
            for f in all_faces:
                known = [s for s in f if s in colors]
                unknown = [s for s in f if s not in colors]
                if len(known) == 3:
                    cols = [colors[s] for s in known]
                    if len(set(cols)) != 3:
                        return "conflict"
                elif len(known) == 2:
                    c0, c1 = colors[known[0]], colors[known[1]]
                    if c0 == c1:
                        return "conflict"
                    col = ({0, 1, 2} - {c0, c1}).pop()
                    if unknown[0] in colors and colors[unknown[0]] != col:
                        return "conflict"
                    if unknown[0] not in colors:
                        colors[unknown[0]] = col
                        pending = True
        n = lx * ly
        if len(colors) < n:
            return "unforced"
        for f in all_faces:
            if len({colors[s] for s in f}) != 3:
                return "conflict"
        return "ok"

    first = all_faces[0]
    outcomes = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        seed = {first[i]: perm[i] for i in range(3)}
        outcomes.append(propagate(seed))
    return {
        "formula_monochrome_edges": len(mono),
        "formula_ok": len(mono) == 0,
        "propagation_outcomes": outcomes,
        "three_coloring_exists": any(o == "ok" for o in outcomes),
        "propagation_fully_forced": all(o != "unforced" for o in outcomes),
    }


def sublattice_flip_commutes(lx: int, ly: int) -> dict:
    """Double-sublattice X products commute with every ZZZ iff every face has one vertex of each color."""
    if not coloring_conflicts(lx, ly)["formula_ok"]:
        # Still count odd overlaps of the formula partition, which is not a coloring.
        colors = {}
        for y in range(ly):
            for x in range(lx):
                colors[site(x, y, lx, ly)] = (x + y) % 3
        odd = 0
        for f in faces(lx, ly)[0] + faces(lx, ly)[1]:
            # flip colors 0 and 1
            hit = sum(1 for s in f if colors[s] in (0, 1))
            if hit % 2:
                odd += 1
        return {"formula_is_coloring": False, "odd_faces_under_formula_flip_01": odd, "commutes": False}
    colors = {}
    for y in range(ly):
        for x in range(lx):
            colors[site(x, y, lx, ly)] = (x + y) % 3
    odd = {pair: 0 for pair in ((0, 1), (0, 2), (1, 2))}
    for f in faces(lx, ly)[0] + faces(lx, ly)[1]:
        cols = [colors[s] for s in f]
        for pair in odd:
            hit = sum(1 for c in cols if c in pair)
            if hit % 2:
                odd[pair] += 1
    return {"formula_is_coloring": True, "odd_faces": {str(k): v for k, v in odd.items()}, "commutes": all(v == 0 for v in odd.values())}


def gamma_period_from_diag(diag: np.ndarray) -> dict:
    scaled = np.rint(diag * 2.0).astype(np.int64)
    if np.max(np.abs(diag - scaled / 2.0)) > 1e-8:
        raise ValueError("diagonal not on the half-integer lattice")
    uniq = np.unique(scaled)
    g = 0
    for value in uniq:
        g = math.gcd(g, int(value - uniq[0]))
    period = 4.0 * math.pi / float(g)
    phase = np.exp(-1j * period * diag)
    spread = float(np.max(np.abs(phase - phase[0])))
    half = np.exp(-1j * 0.5 * period * diag)
    half_spread = float(np.max(np.abs(half / half[0] - 1.0)))
    return {
        "gcd_half_units": int(g),
        "period": period,
        "full_spread": spread,
        "half_spread": half_spread,
        "half_is_not_period": half_spread > 1e-6,
    }

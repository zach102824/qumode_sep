"""Triangular-lattice tori and elementary up/down triangles.

Sites are (x, y) with x = 0..Lx-1, y = 0..Ly-1, periodic.
Index = x + Lx * y. Qubit 0 is not special; the index is only an address.

Neighbor steps of the triangular lattice: (1, 0), (0, 1), (1, 1), and their
reverses. Elementary faces:

    up   ▲(x,y) = {(x,y), (x+1,y), (x+1,y+1)}
    down ▼(x,y) = {(x,y), (x,y+1), (x+1,y+1)}

One of each per plaquette, so M = 2 N when every face is distinct.
A 3-coloring color(x,y) = (x + y) mod 3 is a proper coloring of this graph
only when the periodic images of those steps still change the color, which
requires Lx ≡ 0 (mod 3) and Ly ≡ 0 (mod 3). That is checked, not assumed.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def geometries_for_n(n: int) -> list[tuple[int, int]]:
    """Factorizations Lx * Ly = n with Lx, Ly >= 3, most-square first."""
    pairs = []
    for lx in range(3, int(np.sqrt(n)) + 1):
        if n % lx == 0:
            ly = n // lx
            if ly >= 3:
                pairs.append((lx, ly))
                if ly != lx:
                    pairs.append((ly, lx))
    pairs.sort(key=lambda p: (abs(p[0] - p[1]), p[0]))
    # unique, stable
    seen = set()
    out = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def site_index(x: int, y: int, lx: int, ly: int) -> int:
    return (x % lx) + lx * (y % ly)


def build_triangles(lx: int, ly: int) -> dict:
    n = lx * ly
    up = []
    down = []
    for y in range(ly):
        for x in range(lx):
            up.append(
                tuple(
                    sorted(
                        (
                            site_index(x, y, lx, ly),
                            site_index(x + 1, y, lx, ly),
                            site_index(x + 1, y + 1, lx, ly),
                        )
                    )
                )
            )
            down.append(
                tuple(
                    sorted(
                        (
                            site_index(x, y, lx, ly),
                            site_index(x, y + 1, lx, ly),
                            site_index(x + 1, y + 1, lx, ly),
                        )
                    )
                )
            )
    up_set = set(up)
    down_set = set(down)
    coincidences = sorted(up_set & down_set)
    # edges of the triangular graph
    edge_steps = ((1, 0), (0, 1), (1, 1))
    edges = set()
    for y in range(ly):
        for x in range(lx):
            i = site_index(x, y, lx, ly)
            for dx, dy in edge_steps:
                j = site_index(x + dx, y + dy, lx, ly)
                if i != j:
                    edges.add((min(i, j), max(i, j)))
    return {
        "lx": lx,
        "ly": ly,
        "n": n,
        "up": up,
        "down": down,
        "n_up_listed": len(up),
        "n_down_listed": len(down),
        "n_up_unique": len(up_set),
        "n_down_unique": len(down_set),
        "n_duplicate_up": len(up) - len(up_set),
        "n_duplicate_down": len(down) - len(down_set),
        "n_up_down_coincidences": len(coincidences),
        "coincidence_examples": [list(t) for t in coincidences[:4]],
        "M": len(up_set | down_set),
        "edges": edges,
        "both_divisible_by_3": (lx % 3 == 0) and (ly % 3 == 0),
    }


def color_xy(x: int, y: int) -> int:
    return (x + y) % 3


def coloring_report(lx: int, ly: int, geom: dict) -> dict:
    """Test the formula color = (x+y) mod 3, then try to extend a face coloring."""
    n = lx * ly
    formula = np.array([color_xy(i % lx, i // lx) for i in range(n)], dtype=np.int64)
    bad_edges = []
    for i, j in geom["edges"]:
        if formula[i] == formula[j]:
            bad_edges.append((int(i), int(j)))
    # Each elementary triangle is a K3, so a proper 3-coloring is exactly
    # an assignment of three distinct colors on every face, consistent across faces.
    extended = extend_face_coloring(lx, ly, geom)
    counts = [int(np.sum(formula == c)) for c in range(3)]
    return {
        "formula": "(x + y) mod 3",
        "formula_proper_on_all_edges": len(bad_edges) == 0,
        "n_monochrome_edges": len(bad_edges),
        "monochrome_edge_examples": bad_edges[:6],
        "formula_color_counts": counts,
        "face_coloring_exists": extended["ok"],
        "face_coloring_detail": extended["detail"],
        "face_color_counts": extended.get("counts"),
        "note": (
            "Both periods divisible by 3 is necessary for color=(x+y) mod 3 "
            "to survive the boundary. A failed face extension means this torus "
            "has no 3-sublattice coloring at all."
        ),
    }


def extend_face_coloring(lx: int, ly: int, geom: dict) -> dict:
    """Color by forcing each K3 once two vertices are colored. One global perm."""
    n = geom["n"]
    faces = list(geom["up"]) + list(geom["down"])
    # adjacency of vertices to faces
    color = np.full(n, -1, dtype=np.int64)
    # seed the first face that has 3 distinct vertices
    seed = None
    for f in faces:
        if len(set(f)) == 3:
            seed = f
            break
    if seed is None:
        return {"ok": False, "detail": "no 3-distinct face"}
    color[seed[0]], color[seed[1]], color[seed[2]] = 0, 1, 2
    progress = True
    conflict = None
    guard = 0
    while progress and guard < n * len(faces) + 5:
        guard += 1
        progress = False
        for f in faces:
            cols = [int(color[v]) for v in f]
            known = [(v, c) for v, c in zip(f, cols) if c >= 0]
            unknown = [v for v, c in zip(f, cols) if c < 0]
            if len(known) == 3:
                if len({c for _, c in known}) < 3:
                    conflict = {"face": list(f), "colors": cols}
                    break
            elif len(known) == 2:
                used = {c for _, c in known}
                if len(used) < 2:
                    conflict = {"face": list(f), "colors": cols, "reason": "edge monochrome"}
                    break
                missing = ({0, 1, 2} - used).pop()
                v = unknown[0]
                if color[v] >= 0 and color[v] != missing:
                    conflict = {"face": list(f), "want": missing, "have": int(color[v])}
                    break
                if color[v] < 0:
                    color[v] = missing
                    progress = True
            elif len(known) == 3 and len(unknown) == 0:
                pass
        if conflict is not None:
            break
    n_colored = int(np.sum(color >= 0))
    if conflict is not None:
        return {"ok": False, "detail": {"conflict": conflict, "n_colored": n_colored}}
    if n_colored < n:
        return {"ok": False, "detail": {"uncolored": n - n_colored}}
    # final audit
    for f in faces:
        if len({int(color[v]) for v in f}) < 3:
            return {"ok": False, "detail": {"bad_face": list(f)}}
    counts = [int(np.sum(color == c)) for c in range(3)]
    return {"ok": True, "detail": "consistent", "counts": counts, "colors": color}


def gf2_rank(A: np.ndarray) -> int:
    M = np.array(A, dtype=np.uint8, copy=True) & 1
    rows, cols = M.shape
    rank = 0
    row = 0
    pivots = []
    for col in range(cols):
        pivot = None
        for r in range(row, rows):
            if M[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
        for r in range(rows):
            if r != row and M[r, col]:
                M[r] ^= M[row]
        pivots.append(col)
        rank += 1
        row += 1
        if row == rows:
            break
    return int(rank)


def gf2_solve(A: np.ndarray, b: np.ndarray):
    """Return (solvable, one solution or None, rank)."""
    M = np.array(A, dtype=np.uint8, copy=True) & 1
    rhs = np.array(b, dtype=np.uint8, copy=True).reshape(-1, 1) & 1
    rows, cols = M.shape
    Aug = np.concatenate([M, rhs], axis=1)
    row = 0
    pivot_of_row = [-1] * rows
    for col in range(cols):
        pivot = None
        for r in range(row, rows):
            if Aug[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            Aug[[row, pivot]] = Aug[[pivot, row]]
        for r in range(rows):
            if r != row and Aug[r, col]:
                Aug[r] ^= Aug[row]
        pivot_of_row[row] = col
        row += 1
        if row == rows:
            break
    for r in range(row, rows):
        if Aug[r, cols]:
            return False, None, int(row)
    sol = np.zeros(cols, dtype=np.uint8)
    for r in range(row):
        c = pivot_of_row[r]
        if c >= 0:
            sol[c] = Aug[r, cols]
    return True, sol, int(row)


def left_kernel_basis(A: np.ndarray) -> np.ndarray:
    """Rows that span {y : y^T A = 0} over GF(2). Shape (nullity, n_rows)."""
    M = np.array(A, dtype=np.uint8, copy=True) & 1
    rows, cols = M.shape
    # Row-reduce M itself and record operations on the identity.
    E = np.eye(rows, dtype=np.uint8)
    row = 0
    for col in range(cols):
        pivot = None
        for r in range(row, rows):
            if M[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
            E[[row, pivot]] = E[[pivot, row]]
        for r in range(rows):
            if r != row and M[r, col]:
                M[r] ^= M[row]
                E[r] ^= E[row]
        row += 1
        if row == rows:
            break
    basis = E[row:]
    return basis


def incidence(triangles: list[tuple], n: int) -> np.ndarray:
    A = np.zeros((len(triangles), n), dtype=np.uint8)
    for r, tri in enumerate(triangles):
        for v in tri:
            A[r, int(v)] ^= 1  # xor so a collapsed repeated vertex is not a fake 1
        # proper triangles have 3 distinct vertices; set bits explicitly
    # redo cleanly
    A[:] = 0
    for r, tri in enumerate(triangles):
        for v in set(tri):
            A[r, int(v)] = 1
    return A


def gauge_report(geom: dict) -> dict:
    """Is the up/down sign pattern gauge-equivalent to uniform ferro?

    Uniform ferro cost is -sum_{all faces} ZZZ.
    Staggered λ=1 cost is -sum_▲ ZZZ + sum_▼ ZZZ.
    A gauge ∏_{i in S} X_i flips the sign of each face with odd overlap with S.
    Reaching uniform ferro from staggered λ=1 requires flipping every down
    face and no up face: A x = b with b=0 on ▲ and b=1 on ▼.
    """
    n = geom["n"]
    faces = list(geom["up"]) + list(geom["down"])
    # drop duplicate rows if a torus identified two plaquettes, but keep a
    # parallel b. For the linear system use unique faces with their orientation.
    A = incidence(faces, n)
    b = np.array([0] * len(geom["up"]) + [1] * len(geom["down"]), dtype=np.uint8)
    solvable, sol, rank = gf2_solve(A, b)
    # also the system on UNIQUE faces (if duplicates, the repeated equations must agree)
    uniq = []
    b_u = []
    seen = {}
    consistent_dup = True
    for tri, bit in zip(faces, b):
        if tri in seen and seen[tri] != int(bit):
            consistent_dup = False
        if tri not in seen:
            seen[tri] = int(bit)
            uniq.append(tri)
            b_u.append(int(bit))
    Au = incidence(uniq, n)
    bu = np.array(b_u, dtype=np.uint8)
    solvable_u, sol_u, rank_u = gf2_solve(Au, bu)
    kernel = left_kernel_basis(Au)
    # all-ones dot b
    ones_obstruction = int(np.sum(bu) & 1)  # only meaningful if 1 is in the left kernel
    col_sums = np.sum(Au, axis=0) & 1
    all_ones_in_kernel = bool(np.all(col_sums == 0))
    obstructions = []
    if kernel.size:
        dots = (kernel @ bu.astype(np.uint8)) & 1
        for i, d in enumerate(dots):
            if int(d) == 1:
                weight = int(np.sum(kernel[i]))
                obstructions.append({"kernel_vector": i, "weight": weight})
    return {
        "n_face_rows_with_multiplicity": int(A.shape[0]),
        "rank_with_multiplicity": int(rank),
        "solvable_with_multiplicity": bool(solvable),
        "duplicate_orientations_consistent": consistent_dup,
        "n_unique_faces": len(uniq),
        "rank_unique": int(rank_u),
        "nullity_columns": int(n - rank_u),
        "left_kernel_dim": int(kernel.shape[0]),
        "solvable_unique": bool(solvable_u),
        "gauge_equivalent_to_uniform_ferro": bool(solvable_u),
        "all_ones_in_left_kernel": all_ones_in_kernel,
        "sum_b_mod2": int(np.sum(bu) & 1),
        "all_ones_blocks_staggered": bool(all_ones_in_kernel and (int(np.sum(bu) & 1) == 1)),
        "n_obstructing_kernel_vectors": len(obstructions),
        "lightest_obstruction_weight": min((o["weight"] for o in obstructions), default=None),
        "column_sum_mod2_unique_values": sorted(set(int(x) for x in col_sums)),
    }


def symmetry_operators_report(lx: int, ly: int, geom: dict, coloring: dict) -> dict:
    """Double-sublattice X products commute with every 3-body term iff every
    face has one vertex of each color. That fails when no 3-coloring exists.
    """
    if not coloring["face_coloring_exists"]:
        return {
            "z2xz2_from_sublattice_flips": False,
            "reason": "no 3-face-coloring on this torus, so there are no three sublattices to flip",
        }
    # rebuild colors
    ext = extend_face_coloring(lx, ly, geom)
    colors = ext["colors"]
    groups = {c: np.flatnonzero(colors == c) for c in range(3)}
    # candidate operators: flip two colors
    faces = list(geom["up"]) + list(geom["down"])
    checks = {}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        flipped = set(int(i) for i in groups[a]) | set(int(i) for i in groups[b])
        bad = 0
        for f in faces:
            parity = sum(1 for v in f if v in flipped) & 1
            if parity:
                bad += 1
        checks[f"flip_colors_{a}{b}"] = {"faces_with_odd_overlap": bad, "commutes_with_all_ZZZ": bad == 0}
    return {
        "z2xz2_from_sublattice_flips": all(v["commutes_with_all_ZZZ"] for v in checks.values()),
        "sublattice_sizes": {str(c): int(len(groups[c])) for c in range(3)},
        "operators": checks,
    }

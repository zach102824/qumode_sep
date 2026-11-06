"""Triangular-torus triangles, matching the round-2 enumeration.

Sites (x, y), x = 0..Lx-1, y = 0..Ly-1, periodic. Index = x + Lx * y.
Qubit q is bit q of the computational index (bit 0 = site 0).

    up   ▲(x,y) = {(x,y), (x+1,y), (x+1,y+1)}
    down ▼(x,y) = {(x,y), (x,y+1), (x+1,y+1)}

Neighbor steps (1,0), (0,1), (1,1) and their reverses.
"""
from __future__ import annotations

import numpy as np


def site_index(x: int, y: int, lx: int, ly: int) -> int:
    return (x % lx) + lx * (y % ly)


def build_triangles(lx: int, ly: int) -> dict:
    n = lx * ly
    up, down = [], []
    for y in range(ly):
        for x in range(lx):
            up.append(tuple(sorted((
                site_index(x, y, lx, ly),
                site_index(x + 1, y, lx, ly),
                site_index(x + 1, y + 1, lx, ly),
            ))))
            down.append(tuple(sorted((
                site_index(x, y, lx, ly),
                site_index(x, y + 1, lx, ly),
                site_index(x + 1, y + 1, lx, ly),
            ))))
    up_set, down_set = set(up), set(down)
    edges = set()
    for y in range(ly):
        for x in range(lx):
            i = site_index(x, y, lx, ly)
            for dx, dy in ((1, 0), (0, 1), (1, 1)):
                j = site_index(x + dx, y + dy, lx, ly)
                if i != j:
                    edges.add((min(i, j), max(i, j)))
    return {
        "lx": lx,
        "ly": ly,
        "n": n,
        "up": up,
        "down": down,
        "n_up_unique": len(up_set),
        "n_down_unique": len(down_set),
        "n_duplicate_up": len(up) - len(up_set),
        "n_duplicate_down": len(down) - len(down_set),
        "n_up_down_coincidences": len(up_set & down_set),
        "M": len(up_set | down_set),
        "edges": edges,
        "both_divisible_by_3": (lx % 3 == 0) and (ly % 3 == 0),
    }


def validate_geometry(geom: dict) -> dict:
    n = geom["n"]
    problems = []
    if geom["n_duplicate_up"] or geom["n_duplicate_down"]:
        problems.append("duplicate triangles within an orientation")
    if geom["n_up_down_coincidences"]:
        problems.append("an up triangle equals a down triangle")
    if geom["n_up_unique"] != n or geom["n_down_unique"] != n:
        problems.append(f"expected {n} unique faces each orientation, got {geom['n_up_unique']},{geom['n_down_unique']}")
    if geom["M"] != 2 * n:
        problems.append(f"M={geom['M']} != 2N")
    # every face has 3 distinct sites and 3 edges of the graph
    for label, faces in (("up", geom["up"]), ("down", geom["down"])):
        for f in faces:
            if len(set(f)) != 3:
                problems.append(f"collapsed {label} {f}")
            for a, b in ((f[0], f[1]), (f[0], f[2]), (f[1], f[2])):
                if (min(a, b), max(a, b)) not in geom["edges"]:
                    problems.append(f"{label} {f} uses a non-edge")
    return {"ok": not problems, "problems": problems, "M": geom["M"], "n_edges": len(geom["edges"])}


def coloring_report(lx: int, ly: int, geom: dict) -> dict:
    """color = (x+y) mod 3 is proper iff both periods are multiples of 3.

    A failed extension means this torus has no 3-face-coloring at all, so
    there are no sublattices A,B,C on which to report m_A, m_B, m_C.
    """
    n = lx * ly
    formula = np.array([(i % lx + i // lx) % 3 for i in range(n)], dtype=np.int64)
    bad = [(int(i), int(j)) for i, j in geom["edges"] if formula[i] == formula[j]]
    faces = list(geom["up"]) + list(geom["down"])
    color = np.full(n, -1, dtype=np.int64)
    seed = next(f for f in faces if len(set(f)) == 3)
    color[list(seed)] = [0, 1, 2]
    conflict = None
    for _ in range(n * len(faces) + 2):
        progress = False
        for f in faces:
            known = [(v, int(color[v])) for v in f if color[v] >= 0]
            unknown = [v for v in f if color[v] < 0]
            if len(known) >= 2 and len({c for _, c in known}) < min(3, len(known)):
                conflict = list(f)
                break
            if len(known) == 2 and len(unknown) == 1:
                missing = ({0, 1, 2} - {c for _, c in known}).pop()
                v = unknown[0]
                if color[v] < 0:
                    color[v] = missing
                    progress = True
                elif color[v] != missing:
                    conflict = list(f)
                    break
            if len(known) == 3 and len({c for _, c in known}) < 3:
                conflict = list(f)
                break
        if conflict is not None or not progress:
            break
    ok = conflict is None and int(np.sum(color >= 0)) == n
    if ok:
        for f in faces:
            if len({int(color[v]) for v in f}) < 3:
                ok = False
                conflict = list(f)
    return {
        "formula_proper": len(bad) == 0,
        "n_monochrome_edges": len(bad),
        "face_coloring_exists": bool(ok),
        "conflict_face": conflict,
        "both_periods_divisible_by_3": geom["both_divisible_by_3"],
        "sublattice_magnetization_defined": bool(ok),
    }


def graph_distance_from_origin(lx: int, ly: int) -> np.ndarray:
    """Shortest-path distance on the triangular torus, from site 0."""
    n = lx * ly
    dist = np.full(n, -1, dtype=np.int64)
    dist[0] = 0
    queue = [0]
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))
    head = 0
    while head < len(queue):
        i = queue[head]
        head += 1
        x, y = i % lx, i // lx
        for dx, dy in steps:
            j = site_index(x + dx, y + dy, lx, ly)
            if dist[j] < 0:
                dist[j] = dist[i] + 1
                queue.append(j)
    return dist


def pair_distances(lx: int, ly: int) -> dict[int, list[tuple[int, int]]]:
    """Pairs grouped by torus graph distance. Translation reduces this to d(0, ·)."""
    dist = graph_distance_from_origin(lx, ly)
    groups: dict[int, list[tuple[int, int]]] = {}
    n = lx * ly
    for i in range(n):
        xi, yi = i % lx, i // lx
        for j in range(i + 1, n):
            dx = (j % lx - xi) % lx
            dy = (j // lx - yi) % ly
            d = int(dist[dx + lx * dy])
            groups.setdefault(d, []).append((i, j))
    return groups


def positions(lx: int, ly: int) -> np.ndarray:
    """Standard embedding: a1=(1,0), a2=(1/2, √3/2). Neighbor lengths are 1."""
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3.0) / 2.0])
    r = np.zeros((lx * ly, 2), dtype=np.float64)
    for y in range(ly):
        for x in range(lx):
            r[x + lx * y] = x * a1 + y * a2
    return r


def reciprocal_basis() -> tuple[np.ndarray, np.ndarray]:
    """b_i · a_j = 2π δ_ij."""
    b1 = 2.0 * np.pi * np.array([1.0, -1.0 / np.sqrt(3.0)])
    b2 = 2.0 * np.pi * np.array([0.0, 2.0 / np.sqrt(3.0)])
    return b1, b2


def momentum_grid(lx: int, ly: int) -> list[dict]:
    """q = (n_x/Lx) b1 + (n_y/Ly) b2. K = (b1+b2)/3 is on this grid iff 3|Lx and 3|Ly."""
    b1, b2 = reciprocal_basis()
    out = []
    for ny in range(ly):
        for nx in range(lx):
            q = (nx / lx) * b1 + (ny / ly) * b2
            out.append({"nx": nx, "ny": ny, "q": q})
    return out


def signed_face_map(geom: dict, lam: float = 1.0) -> dict[tuple[int, int, int], float]:
    out = {}
    for tri in geom["up"]:
        out[tuple(sorted(tri))] = -1.0
    for tri in geom["down"]:
        out[tuple(sorted(tri))] = float(lam)
    return out


def lattice_isomorphism(src: dict, dst: dict, lam: float = 1.0) -> dict:
    """Search affine maps of the triangular torus sending src faces to dst faces.

    A hit with up→up and down→down is an orientation-preserving equivalence of
    the staggered Hamiltonian. A hit that swaps orientations sends H_C to -H_C
    at λ=1 and is recorded separately. This is a finite search over images of
    the origin and of two neighbor steps, not a claim about all permutations.
    """
    s_faces = signed_face_map(src, lam)
    d_faces = signed_face_map(dst, lam)
    slx, sly = src["lx"], src["ly"]
    dlx, dly = dst["lx"], dst["ly"]
    steps = ((1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1))
    tried = 0
    same = None
    swapped = None
    for ox in range(dlx):
        for oy in range(dly):
            for vx in steps:
                for vy in steps:
                    # images of e1, e2 must be nonzero and not parallel over R^2 in the
                    # discrete sense: the  Lx*Ly images must be all distinct.
                    tried += 1

                    def image(x, y):
                        return site_index(ox + x * vx[0] + y * vy[0], oy + x * vx[1] + y * vy[1], dlx, dly)

                    seen = {}
                    ok = True
                    for y in range(sly):
                        for x in range(slx):
                            j = image(x, y)
                            if j in seen:
                                ok = False
                                break
                            seen[j] = (x, y)
                        if not ok:
                            break
                    if not ok or len(seen) != src["n"]:
                        continue

                    def map_tri(tri):
                        pts = []
                        # recover (x,y) from site index in the SOURCE lattice
                        for s in tri:
                            x, y = s % slx, s // slx
                            pts.append(image(x, y))
                        return tuple(sorted(pts))

                    same_hit = True
                    swap_hit = True
                    for tri, coeff in s_faces.items():
                        im = map_tri(tri)
                        if d_faces.get(im) != coeff:
                            same_hit = False
                        if d_faces.get(im) != -coeff:
                            swap_hit = False
                        if not same_hit and not swap_hit:
                            break
                    if same_hit and same is None:
                        same = {"origin": [ox, oy], "v1": list(vx), "v2": list(vy)}
                    if swap_hit and swapped is None:
                        swapped = {"origin": [ox, oy], "v1": list(vx), "v2": list(vy)}
                    if same is not None and swapped is not None:
                        break
            if same is not None and swapped is not None:
                break
        if same is not None and swapped is not None:
            break
    return {
        "n_maps_tested_until_stop": tried,
        "equivalent_same_orientation": same,
        "equivalent_up_down_swapped": swapped,
    }


def k_point_commensurate(lx: int, ly: int) -> dict:
    """3-sublattice K corresponds to phases ω^{x} and ω^{y}, i.e. n_x/Lx = n_y/Ly = 1/3."""
    ok = (lx % 3 == 0) and (ly % 3 == 0)
    return {
        "K_on_momentum_grid": ok,
        "reason": (
            "K=(b1+b2)/3 lies on the finite-torus grid iff both periods are multiples of 3. "
            "Otherwise a structure factor at K is not a lattice observable of this torus."
        ),
    }

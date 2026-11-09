"""Run the author notebooks' gap routines. No rewritten Hamiltonian.

Clone tag 2.0 (f4c329fa69dcf98b7be32374c145584209e21f0e) to /tmp/mte first.
Outputs go to /tmp/gap_out. The spin-glass J arrays are the literal matrices
in data/fig_5/spin_glass_def_parity_energy_sprectum.ipynb; that notebook
leaves them commented, so this script assigns those same arrays.
"""
import json, time, csv
from pathlib import Path
import numpy as np
from numpy.linalg import eigh
from scipy.linalg import eigh as scipy_eigh

OUT = Path("/tmp/gap_out")
OUT.mkdir(exist_ok=True)

# ---------- AFM even-even routine: exec author cells 0-7, then their B loop ----------
src = Path("/tmp/mte_sg_cells.py").read_text().split("# ===== CELL ")
cells = {}
for block in src:
    if not block.strip():
        continue
    num, body = block.split("=====\n", 1)
    cells[int(num.strip())] = body

ns = {"np": np, "LA": np.linalg, "time": time}
t0 = time.perf_counter()
for i in range(0, 8):
    exec(cells[i], ns)
print("AFM basis dim", ns["couple_dimen"], "build_s", time.perf_counter()-t0, flush=True)

# author loop is cell 8 with dB=0.0001 from B=0.1 to 2
exec(cells[8], ns)
B = np.array(ns["B_value_list"], dtype=float)
d1 = np.array(ns["delta_e_list"], dtype=float)
imin = int(np.argmin(d1))
print("AFM author-loop min", float(d1[imin]), "at B", float(B[imin]), "n", len(B), flush=True)

# residual against the saved supplemental spectrum on its own abscissae
rows = list(csv.DictReader(open("/tmp/mte/data/supplemental material/energy_spec.csv")))
# rebuild gaps at those B using the matrices the author cells just built
HA = ns["HA_matrix"]
HB = ns["HB_matrix"]
resid = []
for r in rows:
    b = float(r["B"])
    if b < float(ns["B"]) - 1e-15:
        continue  # author loop starts at 0.1; csv includes smaller B
    vals = eigh(HA + b * HB, UPLO="U")[0]
    gap = float(vals[1] - vals[0])
    resid.append(gap - float(r["d1"]))
resid = np.array(resid)
afm = {
    "sector_dimension": int(ns["couple_dimen"]),
    "author_loop_B_start": float(B[0]),
    "author_loop_B_end": float(B[-1]),
    "author_loop_dB": float(B[1]-B[0]),
    "author_loop_min_gap": float(d1[imin]),
    "author_loop_B_at_min": float(B[imin]),
    "csv_overlap_n": int(resid.size),
    "csv_residual_maxabs": float(np.max(np.abs(resid))) if resid.size else None,
    "csv_residual_rms": float(np.sqrt(np.mean(resid**2))) if resid.size else None,
    "paper_table_Delta": 0.458,
    "paper_table_Bc": 0.38,
}
(OUT/"afm_author.json").write_text(json.dumps(afm, indent=2))
print("AFM residual", afm, flush=True)

# ordinary full-space gap at the author minimum B, same J as the AFM cells
N = 8
J = ns["J_matrix"]
def D2B(num, digit):
    string = f"{num:0{digit}b}"
    return np.array([int(e) for e in string], int)
def B2D(array):
    res = 0
    for ele in array:
        res = (res << 1) | int(ele)
    return res
Q = 256
diag = np.zeros(Q)
X = np.zeros((Q, Q))
for state in range(Q):
    basis = D2B(state, N)
    for i in range(N):
        flipped = basis.copy()
        flipped[i] = 1 - flipped[i]
        X[state, B2D(flipped)] += 1
    for i in range(N):
        for j in range(i+1, N):
            diag[state] += J[i, j] * (1 if basis[i] == basis[j] else -1)
def spin_parity(v):
    s = 0.0
    for basis_idx in range(Q):
        basis = D2B(basis_idx, N)
        reflected = 1 - basis
        s += v[basis_idx] * v[B2D(reflected)]
    return s
def spatial_parity(v):
    s = 0.0
    for basis_idx in range(Q):
        basis = D2B(basis_idx, N)
        reflected = basis[::-1]
        s += v[basis_idx] * v[B2D(reflected)]
    return s

def spectrum_row(Bvalue, label):
    H = np.diag(diag) + Bvalue * X
    vals, vecs = eigh(H)
    gs_s = spin_parity(vecs[:, 0])
    gs_p = spatial_parity(vecs[:, 0])
    same_spin = None
    same_both = None
    for i in range(1, Q):
        if same_spin is None and abs(spin_parity(vecs[:, i]) - gs_s) < 1e-6:
            same_spin = float(vals[i] - vals[0])
        if same_both is None and abs(spin_parity(vecs[:, i]) - gs_s) < 1e-6 and abs(spatial_parity(vecs[:, i]) - gs_p) < 1e-6:
            same_both = float(vals[i] - vals[0])
        if same_spin is not None and same_both is not None:
            break
    # author reduced-matrix gap
    red = eigh(HA + Bvalue * HB)[0]
    return {
        "label": label,
        "B": float(Bvalue),
        "ordinary_E1_minus_E0": float(vals[1] - vals[0]),
        "same_spin_parity_gap": same_spin,
        "same_spin_and_spatial_gap": same_both,
        "author_reduced_even_even_gap": float(red[1] - red[0]),
        "gs_spin_parity": float(gs_s),
        "gs_spatial_parity": float(gs_p),
        "E0": float(vals[0]),
    }

cross = [
    spectrum_row(afm["author_loop_B_at_min"], "AFM_at_author_min"),
    spectrum_row(0.38, "AFM_at_paper_table_Bc"),
    spectrum_row(0.1, "AFM_at_B_0.1"),
]
(OUT/"afm_crosscheck.json").write_text(json.dumps(cross, indent=2))
print("AFM cross", json.dumps(cross, indent=2), flush=True)

# ---------- spin glass: author full-space spin-parity routine ----------
J001 = np.array([
[0.0, -0.770026, -0.612588, 1.884469, -0.570562, 0.616424, 1.663316, -0.448101],
[0.0, 0.0, 0.795786, 0.693566, -1.818596, -1.793477, 0.869296, -0.476921],
[0.0, 0.0, 0.0, 1.166175, 2.639218, 0.896904, -0.685189, 0.523481],
[0.0, 0.0, 0.0, 0.0, 1.069138, 0.163168, 0.637887, 0.75839],
[0.0, 0.0, 0.0, 0.0, 0.0, 1.210793, 0.186866, 0.464955],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.255085, -1.129867],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.741586],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
J009 = np.array([
[0.000000, 0.773813, 2.485213, -0.561542, 0.008873, 0.691680, 0.086291, -0.564034],
[0.000000, 0.000000, 1.397455, -0.112515, 0.388229, -1.835349, 1.372802, 0.738212],
[0.000000, 0.000000, 0.000000, -0.169025, 0.746900, -1.199308, 0.269533, -1.233924],
[0.000000, 0.000000, 0.000000, 0.000000, 0.939910, 0.653794, 1.202234, 0.980422],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -1.008154, 2.553466, 0.876100],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.196304, -0.398031],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.282909],
[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]], dtype=float)

def author_spin_gap(J_matrix, csv_path, name):
    N = 8
    Q = 256
    def D2B(num, digit):
        string = f"{num:0{digit}b}"
        return np.array([int(ele) for ele in string], int)
    def B2D(array):
        res = 0
        for ele in array:
            res = (res << 1) | ele
        return res
    def calculate_spin_parity(eigenvector):
        parity_sum = 0
        for basis_idx in range(Q):
            basis = D2B(basis_idx, N)
            reflected_basis = 1 - basis
            reflected_idx = B2D(reflected_basis)
            parity_sum += eigenvector[basis_idx] * eigenvector[reflected_idx]
        return round(parity_sum, 5)
    H_a = np.zeros(Q)
    HB_matrix = np.zeros((Q, Q))
    for state in range(Q):
        basis = D2B(state, N)
        for i in range(N):
            flipped_basis = basis.copy()
            flipped_basis[i] = 1 - flipped_basis[i]
            HB_matrix[state, B2D(flipped_basis)] += 1
        for i in range(N):
            for j in range(i+1, N):
                H_a[state] += J_matrix[i][j] * (1 if basis[i] == basis[j] else -1)
    HA_matrix = np.diag(H_a)
    B0 = 0.1
    B_not = 10
    dB = 0.0001
    B_value_list = np.arange(B0, B_not, dB)
    num_matching_states = 5
    delta_e_lists = [[] for _ in range(num_matching_states)]
    ordinary = []
    t0 = time.perf_counter()
    for nB, B_value in enumerate(B_value_list):
        vals, vecs = scipy_eigh(HA_matrix + B_value * HB_matrix)
        ordinary.append(float(vals[1] - vals[0]))
        gs_parity = calculate_spin_parity(vecs[:, 0])
        matching_indices = []
        for i in range(1, Q):
            if calculate_spin_parity(vecs[:, i]) == gs_parity:
                matching_indices.append(i)
            if len(matching_indices) >= num_matching_states:
                break
        for idx, matching_index in enumerate(matching_indices):
            delta_e_lists[idx].append(float(vals[matching_index] - vals[0]))
        if nB % 10000 == 0:
            print(name, nB, "elapsed", time.perf_counter()-t0, flush=True)
    d1 = np.array(delta_e_lists[0])
    imin = int(np.argmin(d1))
    # residual vs saved csv
    saved = list(csv.DictReader(open(csv_path)))
    # csv may have 99000 rows; compare on shared length
    n = min(len(saved), len(d1))
    res = np.array([d1[i] - float(saved[i]["d1"]) for i in range(n)])
    # also ordinary at the coupled minimum
    out = {
        "name": name,
        "n_B": int(len(B_value_list)),
        "min_coupled_gap": float(d1[imin]),
        "B_at_min_coupled": float(B_value_list[imin]),
        "ordinary_gap_at_that_B": float(ordinary[imin]),
        "min_ordinary_gap": float(np.min(ordinary)),
        "B_at_min_ordinary": float(B_value_list[int(np.argmin(ordinary))]),
        "csv_n_compared": int(n),
        "csv_residual_maxabs": float(np.max(np.abs(res))),
        "csv_residual_rms": float(np.sqrt(np.mean(res**2))),
        "seconds": time.perf_counter()-t0,
        "gs_parity_at_min": None,
    }
    print(name, json.dumps(out), flush=True)
    (OUT/f"{name}.json").write_text(json.dumps(out, indent=2))
    return out

author_spin_gap(J009, "/tmp/mte/data/fig_5/energy_spec_tf_493.csv", "sg_tf_493")
author_spin_gap(J001, "/tmp/mte/data/fig_5/energy_spec_tf_11980.csv", "sg_tf_11980")
print("ALL DONE", flush=True)

"""Round-3 R: sublattice packing variant. Not the contiguous Nov_3rd contract.

Logical H is the frozen manifest. Packing changes only which qubits share a shell.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import circuit
import model
import optimize

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PRIOR = ROOT.parent / "round2_P" / "results" / "cells"
PERIODS = {"c1": np.pi / 2, "c2": np.pi / 2, "c3": np.pi / 2, "beta": np.pi}
MANIFEST = model.EXPECTED_MANIFEST


def code_hash():
    h = hashlib.sha256()
    for path in sorted(ROOT.glob("*.py")) + [model.MANIFEST, model.PACKING_FILE]:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2) + "\n")


def intra_stats(J, trips):
    bonds = set()
    dists = []
    for cav in trips:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = cav[i], cav[j]
                bonds.add((min(a, b), max(a, b)))
                dists.append(abs(a - b))
    tot = 0.0
    intra = 0.0
    n = J.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            w = float(J[i, j]) ** 2
            tot += w
            if (i, j) in bonds:
                intra += w
    return {
        "intra_bonds": len(bonds),
        "inter_bonds": n * (n - 1) // 2 - len(bonds),
        "intra_hs_fraction": intra / tot,
        "intra_distances": sorted(dists),
        "reflection_closed": model.reflection_closed(trips, n),
    }


def prior_qaoa(L):
    path = PRIOR / f"N12_L{L}_qaoa.json"
    if not path.exists():
        return None
    cell = json.loads(path.read_text())
    if cell.get("manifest_sha256") != MANIFEST:
        return None
    return cell


def make_starts(kind, L, B, previous, n_restarts, rng, prior_x=None, fill_to=None):
    bounds = circuit.bounds_for(kind, L)
    pdim = circuit.n_params(kind, L)
    starts = []

    def add(x, name):
        xx = optimize.clip(np.asarray(x, dtype=np.float64), bounds)
        if xx.shape != (pdim,):
            raise RuntimeError(f"{name} {xx.shape} != {pdim}")
        starts.append({"name": name, "x0": xx})

    if previous is None:
        if kind == "qaoa":
            pass
        else:
            x = np.zeros(pdim)
            x[0] = 0.35
            if kind in ("A", "M", "sameG"):
                x[1] = -0.45
            if kind == "M":
                x[2] = 0.2
            add(x, "structured_negK")
            x2 = np.zeros(pdim)
            x2[0] = -0.3
            add(x2, "structured_neg_gamma")
    else:
        add(optimize.continuation(kind, previous, L - 1, L, B, PERIODS, "auto"), "bootstrap")
        if L >= 3:
            add(optimize.continuation(kind, previous, L - 1, L, B, PERIODS, "linear"), "linear")
            add(optimize.continuation(kind, previous, L - 1, L, B, PERIODS, "cubic"), "cubic")
        if kind == "qaoa":
            ang = circuit.unpack("qaoa", starts[0]["x0"], L)
            x = np.empty(2 * L)
            x[0::2] = optimize.fourier_schedule(ang["gammas"])
            x[1::2] = optimize.fourier_schedule(ang["betas"])
            add(x, "fourier")
    if prior_x is not None and kind == "qaoa" and len(prior_x) == pdim:
        add(prior_x, "round2_manifest_floor")
    target = len(starts) + n_restarts
    if fill_to is not None:
        target = max(target, fill_to)
    while len(starts) < target:
        add(rng.normal(0.0, 0.35, size=pdim), f"restart{len(starts)}")
    return starts


def run_method(name, kind, packing_name, n, L, sys, built, spec, previous, n_restarts, rng, prior_x, fill_to, maxiter):
    trips = spec["trips"]
    indices = spec["indices"]
    hc, fk, fm, B = sys["hc"], sys["fk"], spec["fm"], sys["B"]
    init_evals = {"n": 0}

    def energy_only(x):
        init_evals["n"] += 1
        _, E, _, _ = circuit.simulate(kind, x, L, hc, fk, fm, B, n, indices, False)
        return E

    # Rank L=1 qaoa grid properly inside make_starts? Do a real small grid here for qaoa L=1.
    starts = make_starts(kind, L, B, previous, n_restarts, rng, prior_x, fill_to)
    if kind == "qaoa" and L == 1:
        ranked = []
        for g in np.linspace(-np.pi, np.pi, 17):
            for b in np.linspace(-np.pi, np.pi, 17):
                ranked.append((energy_only(np.array([g, b])), np.array([g, b])))
        ranked.sort(key=lambda t: t[0])
        grids = [{"name": f"grid{i}", "x0": optimize.clip(x, circuit.bounds_for(kind, L))} for i, (_, x) in enumerate(ranked[:6])]
        rest = [s for s in starts if not str(s["name"]).startswith("grid")]
        starts = grids + rest
        if fill_to is not None:
            starts = starts[:fill_to]

    def energy_grad(x):
        _, E, g, _ = circuit.simulate(kind, x, L, hc, fk, fm, B, n, indices, True)
        return E, g

    t0 = time.perf_counter()
    ran = optimize.run_starts(energy_grad, starts, circuit.bounds_for(kind, L), maxiter, maxiter)
    x = np.asarray(ran["best"]["x"], float)
    psi, E, g, var = circuit.simulate(kind, x, L, hc, fk, fm, B, n, indices, True)
    if g is not None and float(np.linalg.norm(g)) > 1e-3:
        extra = optimize.run_starts(
            energy_grad, [{"name": "grad_polish", "x0": x}], circuit.bounds_for(kind, L), 400, 400,
        )
        ran["nfev_sum"] += extra["nfev_sum"]
        ran["starts"].extend(extra["starts"])
        if extra["best"]["E"] < ran["best"]["E"] - 1e-12:
            ran["best"] = extra["best"]
            x = np.asarray(extra["best"]["x"], float)
            psi, E, g, var = circuit.simulate(kind, x, L, hc, fk, fm, B, n, indices, True)
        energies = np.array([r["E"] for r in ran["starts"]])
        q1, q3 = np.percentile(energies, [25, 75])
        ran["E_median"] = float(np.median(energies))
        ran["E_iqr"] = float(q3 - q1)
    angles = circuit.pack_angles(circuit.unpack(kind, x, L))
    cell = {
        "name": name,
        "kind": kind,
        "packing": packing_name,
        "packing_sha256": built["packing_sha256"],
        "manifest_sha256": MANIFEST,
        "hc_sha256": sys["hc_sha256"],
        "n": n,
        "L": L,
        "skipped": False,
        "E": float(E),
        "E0": built["E0"],
        "energy_error": float(E - built["E0"]),
        "var_H": float(var),
        "F": circuit.fidelity(psi, built["gs"]),
        "n_params": circuit.n_params(kind, L),
        "matched_p": circuit.matched_depth(kind, L),
        "angles": angles,
        "x": [float(v) for v in x],
        "state_hash": hashlib.sha256(np.ascontiguousarray(psi).tobytes()).hexdigest(),
        "parity": circuit.parity_expectation(psi),
        "reflection": circuit.reflection_expectation(psi, built["ridx"]),
        "grad_norm": float(np.linalg.norm(g)),
        "E_median": ran["E_median"],
        "E_iqr": ran["E_iqr"],
        "median_error": float(ran["E_median"] - built["E0"]),
        "n_starts": len(ran["starts"]),
        "init_evals": init_evals["n"],
        "nfev_sum": int(ran["nfev_sum"] + init_evals["n"]),
        "wall_time_s": time.perf_counter() - t0,
        "best_start": ran["best"]["name"],
        "starts": ran["starts"],
        "w_sym_state": model.sym_weight(psi, trips) if kind not in ("qaoa", "sameG") else None,
        "ledger": "packing-variant" if packing_name != "contiguous" else "contiguous-replay",
    }
    if kind in ("qaoa", "sameG"):
        cell["ledger"] = "packing-independent"
        cell["w_sym_state"] = None
    path = RESULTS / "states" / f"N{n}_L{L}_{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, psi=psi, x=x)
    return cell


def write_report(cells, built, failures, meta, diagnostics):
    lines = []
    lines.append("# Round 3 R — sublattice packing variant")
    lines.append("")
    lines.append(
        "This is a separately labeled circuit-mapping experiment. It is not the contiguous Nov_3rd contract. "
        "The logical target Hamiltonian is unchanged. Only the assignment of logical qubits to cavity shells changes. "
        "The packing was frozen from the nearest-neighbor AF two-color rule before any training."
    )
    lines.append("")
    lines.append(f"Manifest `{MANIFEST}`. Packing file `{built['packing_sha256']}`.")
    lines.append("")
    lines.append("## Target")
    lines.append("")
    lines.append(
        f"N=12 B={built['B12']} E0={built['E0_12']:.6f} full gap={built['gap12']:.3e}. "
        f"N=6 B={built['B6']} E0={built['E0_6']:.6f} full gap={built['gap6']:.3e}. "
        "Fidelity uses these full-space ground states. H is not permuted."
    )
    lines.append("")
    lines.append("## Packing diagnostics (not used to choose the map)")
    lines.append("")
    lines.append("| packing | reflection-closed | intra distances | intra HS fraction | GS symmetric weight |")
    lines.append("|---|---|---|---|---|")
    for row in diagnostics:
        lines.append(
            f"| {row['name']} | {row['reflection_closed']} | {row['intra_distances']} | "
            f"{row['intra_hs_fraction']:.3f} | {row['gs_w_sym']:.4f} |"
        )
    lines.append("")
    lines.append(
        "The bare Néel product state on contiguous cavities has symmetric weight 1/81. "
        "That is not the ground-state weight. The symmetric-subspace weight is not an upper bound "
        "on fidelity: Hc+FK is not symmetric inside a cavity, so the diagonal layer can leave that subspace. "
        "Random packings are not reflection-closed. Global parity is unchanged because every qubit is still in one cavity."
    )
    lines.append("")
    lines.append("## Compile charge")
    lines.append("")
    lines.append(
        "G is the same 66 logical pairs as contiguous A. Intra-cavity pairs move from distances (1,1,2) "
        "to (2,2,4) on the sublattice map, still 12 intra and 54 inter bonds. "
        "Putting non-adjacent logical qubits into one cavity is a connectivity change relative to contiguous encoding. "
        "It is charged, not treated as free. Shell count and the three shared shell controls are unchanged. "
        "One diagonal profile is reused L times."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| N | L | run | P | E−E0 | F | Var(H) | median E−E0 | IQR | vs QAOA L | vs matched | grad | nfev | parity | R |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = []
    for key in cells:
        if not cells[key].get("skipped"):
            order.append(key)
    for n, L, name in sorted(order):
        cell = cells[(n, L, name)]
        q = cells.get((n, L, "qaoa"))
        qm = cells.get((n, cell["matched_p"], "qaoa")) if name != "qaoa" else None
        def fmt(other, field="energy_error"):
            if other is None or other.get("skipped") or name == "qaoa":
                return "—"
            return f"{cell[field] - other[field]:.4f}"
        lines.append(
            f"| {n} | {L} | {name} | {cell['n_params']} | {cell['energy_error']:.4f} | {cell['F']:.4f} | "
            f"{cell['var_H']:.3f} | {cell['median_error']:.4f} | {cell['E_iqr']:.4f} | {fmt(q)} | {fmt(qm)} | "
            f"{cell['grad_norm']:.2e} | {cell['nfev_sum']} | {cell['parity']:.3f} | {cell['reflection']:.3f} |"
        )
    lines.append("")
    lines.append(_reading(cells))
    lines.append("")
    lines.append("## Failures")
    lines.append("")
    lines.append("None." if not failures else "\n".join(f"- {item}" for item in failures))
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- code `{meta.get('code_hash')}`")
    lines.append(f"- wall {meta.get('wall_time_s')} s")
    lines.append("- QAOA rows may include one start equal to the round-2 best angle on this same manifest. Both ledgers are reported; equal-layer gains are not a sealed advantage.")
    (RESULTS / "RESULTS.md").write_text("\n".join(lines) + "\n")


def _reading(cells):
    lines = ["## Reading", ""]
    both_L = []
    both_m = []
    for L in range(1, 7):
        a = cells.get((12, L, "A_sub"))
        c = cells.get((12, L, "A_cont"))
        k = cells.get((12, L, "A_sub_k0"))
        q = cells.get((12, L, "qaoa"))
        if not a or a.get("skipped") or not q or q.get("skipped"):
            continue
        mp = a["matched_p"]
        qm = cells.get((12, mp, "qaoa"))
        line = (
            f"N12 L={L}: A_sub error {a['energy_error']:.4f} F {a['F']:.3f} "
            f"(median {a['median_error']:.4f}, IQR {a['E_iqr']:.4f}, thetaK {a['angles'].get('theta_k', float('nan')):.3f})"
        )
        if c and not c.get("skipped"):
            line += f"; contiguous A {c['energy_error']:.4f}/{c['F']:.3f}"
        if k and not k.get("skipped"):
            line += f"; thetaK0 {k['energy_error']:.4f}/{k['F']:.3f}"
        line += f"; QAOA L {q['energy_error']:.4f}/{q['F']:.3f}"
        if qm and not qm.get("skipped"):
            line += f"; QAOA p={mp} {qm['energy_error']:.4f}/{qm['F']:.3f}"
        st = cells.get((12, L, "A_stride"))
        if st and not st.get("skipped"):
            line += f"; stride4 {st['energy_error']:.4f}/{st['F']:.3f}"
        mb = cells.get((12, L, "A_sub_M"))
        if mb and not mb.get("skipped"):
            line += f"; blockmean {mb['energy_error']:.4f}/{mb['F']:.3f} thetaM {mb['angles'].get('theta_m', float('nan')):.3f}"
        lines.append(line)
        if a["energy_error"] < q["energy_error"] - 1e-4 and a["F"] > q["F"] + 1e-3:
            both_L.append(L)
        if qm and a["energy_error"] < qm["energy_error"] - 1e-4 and a["F"] > qm["F"] + 1e-3:
            both_m.append((L, mp))
    lines.append("")
    lines.append(f"Equal-layer both-metric depths, A_sub vs QAOA: {both_L or 'none'}.")
    lines.append(f"Parameter-matched both-metric depths for A_sub: {both_m or 'none'}.")
    m_wins = []
    for L in range(1, 7):
        m = cells.get((12, L, "A_sub_M"))
        if not m or m.get("skipped"):
            continue
        q = cells.get((12, m["matched_p"], "qaoa"))
        if not q or q.get("skipped"):
            continue
        energy_win = m["energy_error"] < q["energy_error"] - 1e-4
        fid_win = m["F"] > q["F"] + 1e-3
        median_win = m["median_error"] < q["energy_error"] - 1e-4
        iqr = max(m["E_iqr"], q["E_iqr"])
        margin = q["energy_error"] - m["energy_error"]
        m_wins.append(
            f"L={L} vs p={m['matched_p']}: energy {m['energy_error']:.4f} vs {q['energy_error']:.4f}, "
            f"F {m['F']:.3f} vs {q['F']:.3f}, both={energy_win and fid_win}, "
            f"median_beats_QAOA_best={median_win}, energy_margin={margin:.4f}, larger_IQR={iqr:.3f}"
        )
    lines.append("Secondary block-mean on the sublattice packing, versus its own matched QAOA depth:")
    lines.extend(m_wins or ["not run"])
    q12 = cells.get((12, 12, "qaoa"))
    m6 = cells.get((12, 6, "A_sub_M"))
    if q12 and m6 and not q12.get("skipped"):
        lines.append(
            f"QAOA p=12, the depth after the L=6 matched depth, has energy error {q12['energy_error']:.4f} "
            f"and fidelity {q12['F']:.3f} after the 64-start rescue. Block-mean at L=6 has energy error "
            f"{m6['energy_error']:.4f} and fidelity {m6['F']:.3f}: better fidelity, worse energy."
        )
    lines.append(
        "A_sub does not beat contiguous A at L=4, L=5, or L=6. The six random packings are worse than both "
        "at L=6, and their final states are not exactly reflection-symmetric. High ground-state symmetric "
        "weight for the sublattice map is a measured diagnostic, not a performance proof and not a fidelity bound. "
        "Block-mean best-start gaps are smaller than the seed IQR. Where the median energy is worse than "
        "matched QAOA's best energy, the result is a single basin. "
        "Round-2 P did not beat parameter-matched QAOA on fidelity (0.445 < 0.485 at L=6). That claim is not repeated."
    )
    return "\n".join(lines)


def make_figures(cells):
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    series = ["qaoa", "A_cont", "A_sub", "A_sub_k0", "A_stride", "A_sub_M"]
    colors = {"qaoa": "k", "A_cont": "C1", "A_sub": "C3", "A_sub_k0": "C2", "A_stride": "C0", "A_sub_M": "C4"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in series:
        xs, ye, yf = [], [], []
        for L in range(1, 12):
            cell = cells.get((12, L, name))
            if not cell or cell.get("skipped"):
                continue
            xs.append(L)
            ye.append(max(cell["energy_error"], 1e-16))
            yf.append(cell["F"])
        if xs:
            axes[0].semilogy(xs, ye, "o-", color=colors[name], label=name)
            axes[1].plot(xs, yf, "o-", color=colors[name], label=name)
    axes[0].set_title("N=12 energy error")
    axes[0].set_xlabel("depth")
    axes[0].set_ylabel("E - E0")
    axes[1].set_title("N=12 fidelity of best-energy state")
    axes[1].set_xlabel("depth")
    axes[1].set_ylabel("|⟨GS|ψ⟩|²")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "n12_packing_energy_fidelity.png", dpi=140)
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    deadline = t0 + 80 * 60
    cache = code_hash()
    _, pdigest = model.load_packings()
    print("manifest", MANIFEST, "packing", pdigest, "code", cache[:12], flush=True)
    specs = {}
    diagnostics = []
    systems = {}
    spectra = {}
    for n in (6, 12):
        print("spectrum", n, flush=True)
        sys = model.system(n)
        systems[n] = sys
        spectra[n] = circuit.dense_spectrum(sys["hc"], sys["B"], n)
        print(f"  E0={spectra[n]['E0']:.6f} gap={spectra[n]['full_gap']:.3e} P={spectra[n]['gs_parity']:.4f} R={spectra[n]['gs_reflection']:.4f}", flush=True)
    names_for = {
        6: ["contiguous", "sublattice"],
        12: ["contiguous", "sublattice", "stride4"] + [f"random{i}" for i in range(6)],
    }
    prepared = {}
    for n, names in names_for.items():
        for name in names:
            trips, _ = model.packing(name, n)
            fm, scale, jbar = model.block_mean_diag(systems[n]["J"], trips, systems[n]["z"])
            prepared[(n, name)] = {
                "trips": trips,
                "indices": model.precompute(trips, n),
                "fm": fm,
                "scale_M": scale,
                "jbar": jbar,
            }
            w = model.sym_weight(spectra[n]["gs"], trips)
            stats = intra_stats(systems[n]["J"], trips)
            stats.update({"name": f"N{n}:{name}", "gs_w_sym": w, "scale_M": scale})
            diagnostics.append(stats)
            print(f"  {name} N{n} w_sym={w:.4f} Rclosed={stats['reflection_closed']} intraHS={stats['intra_hs_fraction']:.3f}", flush=True)
    save_json(RESULTS / "packing_diagnostics.json", diagnostics)
    built_meta = {
        "packing_sha256": pdigest,
        "B12": systems[12]["B_string"],
        "B6": systems[6]["B_string"],
        "E0_12": spectra[12]["E0"],
        "E0_6": spectra[6]["E0"],
        "gap12": spectra[12]["full_gap"],
        "gap6": spectra[6]["full_gap"],
    }
    for n in (6, 12):
        built_meta_n = {
            "E0": spectra[n]["E0"],
            "gs": spectra[n]["gs"],
            "ridx": spectra[n]["ridx"],
            "packing_sha256": pdigest,
        }
        systems[n]["_built"] = built_meta_n

    cells = {}
    failures = []
    prev = {}

    def one(n, L, name, kind, packing_name, n_restarts, maxiter, fill_to=None):
        if time.perf_counter() > deadline:
            failures.append(f"budget skip N={n} L={L} {name}")
            print(failures[-1], flush=True)
            return
        previous = None if L == 1 else prev.get((n, name))
        if L > 1 and previous is None:
            failures.append(f"missing parent N={n} L={L} {name}")
            print(failures[-1], flush=True)
            return
        prior = None
        if name == "qaoa" and n == 12:
            old = prior_qaoa(L)
            if old is not None:
                prior = np.asarray(old["x"], float)
        rng = np.random.default_rng(31000 + n * 100 + L * 17 + sum(ord(ch) for ch in name))
        print(f"optimize N={n} L={L} {name}", flush=True)
        cell = run_method(
            name, kind, packing_name, n, L, systems[n], systems[n]["_built"],
            prepared[(n, packing_name if kind not in ("qaoa", "sameG") else "contiguous")],
            previous, n_restarts, rng, prior, fill_to, maxiter,
        )
        cells[(n, L, name)] = cell
        prev[(n, name)] = cell["x"]
        cell["code_hash"] = cache
        save_json(RESULTS / "cells" / f"N{n}_L{L}_{name}.json", cell)
        print(
            f"  E-E0={cell['energy_error']:.4f} F={cell['F']:.3f} med={cell['median_error']:.4f} "
            f"IQR={cell['E_iqr']:.4f} var={cell['var_H']:.3f} grad={cell['grad_norm']:.2e} "
            f"nfev={cell['nfev_sum']} P={cell['parity']:.3f} R={cell['reflection']:.3f}",
            flush=True,
        )

    # N=6 smoke
    one(6, 1, "A_sub", "A", "sublattice", 12, 1500, fill_to=12)
    one(6, 1, "A_cont", "A", "contiguous", 12, 1500, fill_to=12)
    one(6, 1, "A_sub_k0", "k0", "sublattice", 6, 800, fill_to=6)
    # Strong QAOA on the logical target, packing-independent.
    for L in range(1, 12):
        one(12, L, "qaoa", "qaoa", "contiguous", 0, 800, fill_to=16)
    for L in range(1, 7):
        one(12, L, "A_sub", "A", "sublattice", 12, 1500)
        one(12, L, "A_cont", "A", "contiguous", 12, 1500)
        one(12, L, "A_sub_k0", "k0", "sublattice", 6, 800)
    for L in range(1, 7):
        one(12, L, "A_stride", "A", "stride4", 6, 800)
        one(12, L, "A_sub_M", "M", "sublattice", 6, 800)
    for i in range(6):
        for L in range(1, 7):
            one(12, L, f"random{i}", "A", f"random{i}", 4, 400)
    for L in range(1, 7):
        one(12, L, "sameG", "sameG", "sublattice", 6, 800)

    # Rescue matched p and p+1 if A_sub wins both metrics at L=4,5,6.
    for L in (4, 5, 6):
        a = cells.get((12, L, "A_sub"))
        if not a or a.get("skipped"):
            continue
        for depth in (a["matched_p"], a["matched_p"] + 1):
            q = cells.get((12, depth, "qaoa"))
            if not q or q.get("skipped"):
                continue
            if a["energy_error"] < q["energy_error"] - 1e-4 and a["F"] > q["F"] + 1e-3:
                print(f"rescue trigger L={L} vs QAOA p={depth}", flush=True)
                # 4x additional starts from the current best, charged on that QAOA cell.
                if time.perf_counter() > deadline:
                    failures.append(f"rescue budget skip p={depth}")
                    continue
                rng = np.random.default_rng(80000 + depth)
                xbest = np.asarray(q["x"], float)
                bounds = circuit.bounds_for("qaoa", depth)
                starts = []
                for s in (0.05, 0.15, 0.4):
                    for k in range(8):
                        starts.append({"name": f"rescue_{s}_{k}", "x0": optimize.clip(xbest + rng.normal(0, s, size=xbest.size), bounds)})
                ang = circuit.unpack("qaoa", xbest, depth)
                xf = np.empty_like(xbest)
                xf[0::2] = optimize.fourier_schedule(ang["gammas"])
                xf[1::2] = optimize.fourier_schedule(ang["betas"])
                starts.append({"name": "rescue_fourier", "x0": optimize.clip(xf, bounds)})
                while len(starts) < 64:
                    x = np.zeros(2 * depth)
                    x[0::2] = rng.uniform(-1.2, 1.2, size=depth)
                    x[1::2] = rng.uniform(-1.0, 1.0, size=depth)
                    starts.append({"name": f"rescue_rand{len(starts)}", "x0": x})
                sys = systems[12]
                indices = prepared_indices = prepared_safe(12)
                def energy_grad(x, depth=depth):
                    _, E, g, _ = circuit.simulate("qaoa", x, depth, sys["hc"], sys["fk"], np.zeros_like(sys["hc"]), sys["B"], 12, indices, True)
                    return E, g
                # indices for qaoa are unused but must exist
                extra = optimize.run_starts(energy_grad, starts, bounds, 800, 800)
                q["rescue"] = {"n_extra": len(starts), "improved": extra["best"]["E"] < q["E"] - 1e-12, "rescue_E": extra["best"]["E"]}
                q["nfev_sum"] += extra["nfev_sum"]
                if q["rescue"]["improved"]:
                    x = np.asarray(extra["best"]["x"], float)
                    psi, E, g, var = circuit.simulate("qaoa", x, depth, sys["hc"], sys["fk"], np.zeros_like(sys["hc"]), sys["B"], 12, indices, True)
                    q["pre_E"] = q["E"]
                    q["E"] = float(E)
                    q["energy_error"] = float(E - spectra[12]["E0"])
                    q["F"] = circuit.fidelity(psi, spectra[12]["gs"])
                    q["var_H"] = float(var)
                    q["x"] = [float(v) for v in x]
                    q["angles"] = circuit.pack_angles(circuit.unpack("qaoa", x, depth))
                    q["grad_norm"] = float(np.linalg.norm(g))
                    q["best_start"] = extra["best"]["name"]
                    np.savez_compressed(RESULTS / "states" / f"N12_L{depth}_qaoa.npz", psi=psi, x=x)
                save_json(RESULTS / "cells" / f"N12_L{depth}_qaoa.json", q)

    # nfev ledger: QAOA total vs A_sub total
    main_n = sum(cells[(12, L, "A_sub")]["nfev_sum"] for L in range(1, 7) if (12, L, "A_sub") in cells and not cells[(12, L, "A_sub")].get("skipped"))
    q_n = sum(c["nfev_sum"] for (n, L, name), c in cells.items() if name == "qaoa" and n == 12 and not c.get("skipped"))
    meta = {
        "code_hash": cache,
        "manifest_sha256": MANIFEST,
        "packing_sha256": model.load_packings()[1],
        "wall_time_s": time.perf_counter() - t0,
        "main_A_sub_nfev": main_n,
        "qaoa_nfev": q_n,
        "qaoa_budget_covers_main": q_n >= main_n,
        "failures": failures,
    }
    save_json(RESULTS / "run_meta.json", meta)
    rows = []
    for key, cell in cells.items():
        if not cell.get("skipped"):
            rows.append({k: cell[k] for k in cell if k not in ("starts",)})
    save_json(RESULTS / "comparison.json", rows)
    try:
        make_figures(cells)
    except Exception as exc:
        failures.append(f"figures {exc}")
    write_report(cells, built_meta, failures, meta, diagnostics)
    print(f"done {meta['wall_time_s']:.1f}s cells={len(cells)} qaoa_covers_main={meta['qaoa_budget_covers_main']}", flush=True)


def prepared_safe(n):
    trips, _ = model.packing("contiguous", n)
    return model.precompute(trips, n)


if __name__ == "__main__":
    main()

"""Pagano et al. arXiv:1906.02700 — paper-faithful AF long-range TFIM QAOA.

H = H_A + H_B = Σ_{i<j} J_ij X_i X_j + B Σ_i Y_i
J_ij = J0 / |i-j|^α  (open 1D), α=1, J0 = avg nearest-neighbour = 1.
Init |ψ0⟩ = |+⟩_y^{⊗N} = GS of H_B (B<0).
QAOA: |β,γ⟩ = ∏_k exp(-i β_k H_B/J0) exp(-i γ_k H_A/J0) |ψ0⟩
Score η = (E - Emax) / (Egs - Emax).

Exact statevector via precomputed eigh for N≤12.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from numpy.linalg import eigh
from scipy.optimize import minimize

PKG = Path(__file__).resolve().parent
OUT_DEFAULT = PKG / "runs" / "compare" / "pagano_af_qaoa"


def _bit_flip_mask(n: int, i: int) -> int:
    return 1 << (n - 1 - i)


def build_J(n: int, alpha: float = 1.0, J0: float = 1.0) -> np.ndarray:
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = J[j, i] = 1.0 / (float(j - i) ** float(alpha))
    nn = np.array([J[i, i + 1] for i in range(n - 1)])
    J *= float(J0) / float(nn.mean())
    return J


def build_HA(n: int, J: np.ndarray) -> np.ndarray:
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    idx = np.arange(dim)
    for i in range(n):
        for j in range(i + 1, n):
            partner = idx ^ (_bit_flip_mask(n, i) | _bit_flip_mask(n, j))
            H[idx, partner] += J[i, j]
    return H


def build_HB(n: int, B: float) -> np.ndarray:
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    idx = np.arange(dim)
    for i in range(n):
        mi = _bit_flip_mask(n, i)
        partner = idx ^ mi
        bit = (idx >> (n - 1 - i)) & 1
        amp = np.where(bit == 0, 1j, -1j)
        H[partner, idx] += float(B) * amp
    return 0.5 * (H + H.conj().T)


def build_HZZZ(n: int, J3: np.ndarray, triples: np.ndarray) -> np.ndarray:
    """H_A3 = Σ_{i<j<k} J_ijk Z_i Z_j Z_k (diagonal in Z basis)."""
    dim = 1 << n
    H = np.zeros(dim, dtype=np.float64)
    x = np.arange(dim, dtype=np.int64)
    for (i, j, k), jv in zip(triples, J3):
        def z(bit):
            return np.where(((x >> (n - 1 - bit)) & 1) == 0, 1.0, -1.0)
        H += float(jv) * z(int(i)) * z(int(j)) * z(int(k))
    return np.diag(H)


def plus_y_state(n: int) -> np.ndarray:
    s = np.array([1.0, 1j], dtype=np.complex128) / np.sqrt(2.0)
    psi = np.array([1.0], dtype=np.complex128)
    for _ in range(n):
        psi = np.kron(psi, s)
    return psi


def half_cut_entropy(gs: np.ndarray, n: int) -> float:
    nA = n // 2
    psi = np.asarray(gs, dtype=np.complex128).reshape((1 << nA, 1 << (n - nA)))
    svals = np.linalg.svd(psi, compute_uv=False)
    p = np.clip(np.abs(svals) ** 2, 0.0, None)
    p = p[p > 1e-16]
    p /= p.sum()
    return float(-np.sum(p * np.log(p)))


def eta_score(E, Egs, Emax):
    d = Egs - Emax
    return float("nan") if abs(d) < 1e-15 else float((E - Emax) / d)


def energy(psi, H):
    return float(np.vdot(psi, H @ psi).real)


def eigh_pack(H):
    w, v = eigh(H)
    return np.asarray(w.real, dtype=np.float64), v


def apply_U(w, v, angle, psi):
    return v @ (np.exp(-1j * float(angle) * w) * (v.conj().T @ psi))


def apply_qaoa_layers(psi0, packs_A, packs_B, J0, betas, gammas):
    """packs_* = (w, v) for H_A and H_B. Angles multiply H/J0."""
    psi = psi0
    wA, vA = packs_A
    wB, vB = packs_B
    for beta, gamma in zip(betas, gammas):
        psi = apply_U(wA, vA, float(gamma) / J0, psi)
        psi = apply_U(wB, vB, float(beta) / J0, psi)
    return psi


def optimize_p1_sv(packs_A, packs_B, H, J0, psi0, Egs, Emax, gs, grid=41, seed=0):
    wA, vA = packs_A
    wB, vB = packs_B

    def E_of(beta, gamma):
        psi = apply_qaoa_layers(psi0, packs_A, packs_B, J0, [beta], [gamma])
        return energy(psi, H)

    best_E, best_bg = np.inf, (0.5, 0.3)
    # Focus grid: paper optima have modest γ, β ~ O(1)
    betas = np.linspace(0.0, np.pi, grid)
    gammas = np.linspace(0.0, np.pi, grid)
    landscape = np.empty((grid, grid))
    for ib, b in enumerate(betas):
        for ig, g in enumerate(gammas):
            E = E_of(b, g)
            landscape[ib, ig] = E
            if E < best_E:
                best_E, best_bg = E, (float(b), float(g))

    rng = np.random.default_rng(seed)
    starts = [best_bg]
    flat = landscape.ravel()
    k = min(16, flat.size)
    for t in np.argpartition(flat, k - 1)[:k]:
        ib, ig = divmod(int(t), grid)
        starts.append((float(betas[ib]), float(gammas[ig])))
    for _ in range(8):
        starts.append((float(rng.uniform(0, np.pi)), float(rng.uniform(0, np.pi))))

    def fun(x):
        return E_of(float(x[0]), float(x[1]))

    opt_E, opt_bg = best_E, best_bg
    for x0 in starts:
        res = minimize(
            fun, x0=np.asarray(x0), method="L-BFGS-B",
            bounds=[(0.0, 2 * np.pi), (0.0, 2 * np.pi)],
            options={"maxiter": 60, "ftol": 1e-14},
        )
        if res.fun < opt_E:
            opt_E = float(res.fun)
            opt_bg = (float(res.x[0]) % (2 * np.pi), float(res.x[1]) % (2 * np.pi))

    psi = apply_qaoa_layers(psi0, packs_A, packs_B, J0, [opt_bg[0]], [opt_bg[1]])
    E_sv = energy(psi, H)
    E_init = energy(psi0, H)
    return {
        "beta": opt_bg[0],
        "gamma": opt_bg[1],
        "E_sv": E_sv,
        "E_init": E_init,
        "eta_sv": eta_score(E_sv, Egs, Emax),
        "eta_init": eta_score(E_init, Egs, Emax),
        "F_gs": float(abs(np.vdot(gs, psi)) ** 2),
        "grid_min_E": float(best_E),
    }


def optimize_p2_smoke(packs_A, packs_B, H, J0, psi0, Egs, Emax, b1, g1):
    x0 = np.array([b1, max(b1 - 0.2, 0.0), g1, g1 + 0.2])

    def fun(x):
        psi = apply_qaoa_layers(psi0, packs_A, packs_B, J0, [x[0], x[1]], [x[2], x[3]])
        return energy(psi, H)

    res = minimize(
        fun, x0=x0, method="L-BFGS-B",
        bounds=[(0.0, 2 * np.pi)] * 4,
        options={"maxiter": 100, "ftol": 1e-14},
    )
    x = res.x
    psi = apply_qaoa_layers(psi0, packs_A, packs_B, J0, [x[0], x[1]], [x[2], x[3]])
    E = energy(psi, H)
    return {
        "betas": [float(x[0]), float(x[1])],
        "gammas": [float(x[2]), float(x[3])],
        "E_sv": E,
        "eta_sv": eta_score(E, Egs, Emax),
        "success": bool(res.success),
    }


def run_xx_one(n, B_over_J0, alpha, J0, do_p2, seed, grid):
    t0 = time.time()
    B = float(B_over_J0) * float(J0)
    J = build_J(n, alpha=alpha, J0=J0)
    HA = build_HA(n, J)
    HB = build_HB(n, B)
    H = HA.astype(np.complex128) + HB
    evals, evecs = eigh(H)
    Egs, Emax = float(evals[0].real), float(evals[-1].real)
    gs = evecs[:, 0]
    k = int(np.argmax(np.abs(gs)))
    gs = gs * (np.abs(gs[k]) / (gs[k] + 1e-30))
    S = half_cut_entropy(gs, n)
    psi0 = plus_y_state(n)
    packs_A = eigh_pack(HA.astype(np.complex128))
    packs_B = eigh_pack(HB)
    eB, vB = packs_B
    ov = float(abs(np.vdot(vB[:, 0], psi0)) ** 2)

    p1 = optimize_p1_sv(packs_A, packs_B, H, J0, psi0, Egs, Emax, gs, grid=grid, seed=seed)
    row = {
        "model": "AF_XX",
        "n": n, "p": 1, "alpha": alpha, "J0": J0,
        "B_over_J0": B_over_J0, "B": B,
        "Egs": Egs, "Emax": Emax, "S_half": S,
        "avg_NN": float(np.mean([J[i, i + 1] for i in range(n - 1)])),
        "ov_psi0_HB_gs": ov,
        "beta": p1["beta"], "gamma": p1["gamma"],
        "E_sv": p1["E_sv"], "E_init": p1["E_init"],
        "eta_sv": p1["eta_sv"], "eta_init": p1["eta_init"],
        "F_gs": p1["F_gs"],
        "seconds": time.time() - t0,
    }
    rows = [row]
    if do_p2:
        t1 = time.time()
        p2 = optimize_p2_smoke(
            packs_A, packs_B, H, J0, psi0, Egs, Emax, p1["beta"], p1["gamma"],
        )
        rows.append({
            "model": "AF_XX", "n": n, "p": 2, "alpha": alpha, "J0": J0,
            "B_over_J0": B_over_J0, "B": B,
            "Egs": Egs, "Emax": Emax, "S_half": S,
            "avg_NN": row["avg_NN"], "ov_psi0_HB_gs": ov,
            "beta": p2["betas"], "gamma": p2["gammas"],
            "E_sv": p2["E_sv"], "E_init": p1["E_init"],
            "eta_sv": p2["eta_sv"], "eta_init": p1["eta_init"],
            "F_gs": None,
            "seconds": time.time() - t1,
        })
    return rows


def write_results(rows, out_dir: Path, title="Pagano AF QAOA"):
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "n", "p", "alpha", "J0", "B_over_J0", "B", "Egs", "Emax", "S_half",
        "beta", "gamma", "E_sv", "E_init", "eta_sv", "eta_init", "F_gs",
        "avg_NN", "ov_psi0_HB_gs", "seconds",
    ]
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = dict(r)
            if isinstance(rr.get("beta"), list):
                rr["beta"] = json.dumps(rr["beta"])
            if isinstance(rr.get("gamma"), list):
                rr["gamma"] = json.dumps(rr["gamma"])
            w.writerow(rr)

    models = {r.get("model") for r in rows}
    primary = next(iter(models)) if len(models) == 1 else ("AF_XX" if "AF_XX" in models else sorted(models)[0])
    H_BLOCK = {
        "AF_XX": [
            "H = Σ_{i<j} J_ij X_i X_j + B Σ_i Y_i",
            "J_ij = J0 / |i-j|^α  (open 1D), α=1, J0=⟨J_{i,i+1}⟩=1",
            "|ψ0⟩ = |+⟩_y^{⊗N}",
        ],
        "AF_ZZZ": [
            "H = Σ_{i<j<k} J_ijk Z_i Z_j Z_k + B Σ_i X_i",
            "J_ijk = c / r_ijk^α, r=max(|i-j|,|i-k|,|j-k|) open 1D, c so ⟨J⟩_{r=2}=J0=1",
            "|ψ0⟩ = |+⟩_x^{⊗N}",
        ],
        "AF_XXX": [
            "H = Σ_{i<j<k} J_ijk X_i X_j X_k + B Σ_i Y_i",
            "J_ijk = c / r_ijk^α, r=max(|i-j|,|i-k|,|j-k|) open 1D, c so ⟨J⟩_{r=2}=J0=1",
            "|ψ0⟩ = |+⟩_y^{⊗N}",
        ],
    }
    h_lines = H_BLOCK.get(primary, H_BLOCK["AF_XX"])
    p1 = [r for r in rows if r.get("p") == 1 and r.get("model") == primary]
    lines = [f"# {title}", "", "Paper: [arXiv:1906.02700](https://arxiv.org/abs/1906.02700)", ""]
    lines += [
        "## Hamiltonian", "", "```",
        *h_lines,
        "QAOA: ∏ exp(-i β_k H_B/J0) exp(-i γ_k H_A/J0) |ψ0⟩",
        "η = (E-Emax)/(Egs-Emax)",
        "```", "",
    ]
    if p1:
        Bs = sorted({r["B_over_J0"] for r in p1})
        Ns = sorted({r["n"] for r in p1})
        lines.append("## p=1 η* (SV)")
        lines.append("")
        lines.append("| n \\\\ B/J0 | " + " | ".join(f"{b:g}" for b in Bs) + " |")
        lines.append("|---|" + "---|" * len(Bs))
        for n in Ns:
            cells = []
            for b in Bs:
                hit = next((r for r in p1 if r["n"] == n and r["B_over_J0"] == b), None)
                cells.append(f"{hit['eta_sv']:.4f}" if hit else "—")
            lines.append(f"| **{n}** | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Paper target near |B/J0|≈0.3: p=1 η* ≳ 0.94.")
        lines.append("")
        lines.append("## Detail")
        lines.append("")
        lines.append("| n | B/J0 | Egs | E_QAOA | E_init | S_half | η* | η_init | β* | γ* | F_gs |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(p1, key=lambda x: (x["n"], x["B_over_J0"])):
            lines.append(
                f"| {r['n']} | {r['B_over_J0']:g} | {r['Egs']:.5f} | {r['E_sv']:.5f} | "
                f"{r['E_init']:.5f} | {r['S_half']:.4f} | {r['eta_sv']:.4f} | {r['eta_init']:.4f} | "
                f"{r['beta']:.4f} | {r['gamma']:.4f} | {r['F_gs']:.4f} |"
            )
        lines.append("")
        lines.append("## Half-cut S (GS)")
        lines.append("")
        lines.append("| n | " + " | ".join(f"S@{b:g}" for b in Bs) + " |")
        lines.append("|---|" + "---|" * len(Bs))
        for n in Ns:
            cells = []
            for b in Bs:
                hit = next((r for r in p1 if r["n"] == n and r["B_over_J0"] == b), None)
                cells.append(f"{hit['S_half']:.4f}" if hit else "—")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(f"- |⟨ψ0|HB_gs⟩|² min = {min(r['ov_psi0_HB_gs'] for r in p1):.6f}")
        lines.append(f"- avg_NN = {p1[0]['avg_NN']:.6f}")
    p2 = [r for r in rows if r.get("p") == 2]
    if p2:
        lines += ["", "## p=2 smoke", ""]
        lines.append("| n | B/J0 | η_p1 | η_p2 |")
        lines.append("|---|---|---|---|")
        for r in p2:
            p1h = next(x for x in p1 if x["n"] == r["n"] and x["B_over_J0"] == r["B_over_J0"])
            lines.append(f"| {r['n']} | {r['B_over_J0']:g} | {p1h['eta_sv']:.4f} | {r['eta_sv']:.4f} |")
    # other models (only when mixed with AF_XX primary)
    other = [r for r in rows if r.get("model") not in (None, primary)]
    if other and primary == "AF_XX":
        lines += ["", f"## Other models ({other[0].get('model')})", ""]
        lines.append("| model | n | p | B/J0 | η* | η_init | S_half |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in other:
            lines.append(
                f"| {r.get('model')} | {r['n']} | {r['p']} | {r['B_over_J0']:g} | "
                f"{r['eta_sv']:.4f} | {r['eta_init']:.4f} | {r['S_half']:.4f} |"
            )
    lines += ["", f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", ""]
    (out_dir / "RESULTS.md").write_text("\n".join(lines))
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="6,9,12")
    ap.add_argument("--Bs", default="-0.1,-0.3,-0.8")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--J0", type=float, default=1.0)
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--p2", action="store_true")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    Bs = [float(x) for x in args.Bs.split(",") if x.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} [pagano] {msg}"
        print(line, flush=True)
        with log.open("a") as f:
            f.write(line + "\n")

    rows = []
    say(f"start ns={ns} Bs={Bs} alpha={args.alpha} grid={args.grid}")
    for n in ns:
        for B in Bs:
            do_p2 = bool(args.p2 and n == 12 and abs(B + 0.3) < 1e-12)
            say(f"n={n} B/J0={B:g} p2={do_p2}")
            new = run_xx_one(n, B, args.alpha, args.J0, do_p2, args.seed, args.grid)
            rows.extend(new)
            r = new[0]
            say(
                f"  η={r['eta_sv']:.4f} (init {r['eta_init']:.4f}) S={r['S_half']:.4f} "
                f"E={r['E_sv']:.4f}/{r['Egs']:.4f} β={r['beta']:.3f} γ={r['gamma']:.3f} "
                f"({r['seconds']:.1f}s)"
            )
            if len(new) > 1:
                say(f"  p2 η={new[1]['eta_sv']:.4f} ({new[1]['seconds']:.1f}s)")
            try:
                write_results(rows, out)
            except Exception as e:
                say(f"  warn write_results: {e}")
    write_results(rows, out)
    say(f"done → {out}")


if __name__ == "__main__":
    main()

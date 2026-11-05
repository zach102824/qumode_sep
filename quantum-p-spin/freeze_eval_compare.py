"""Freeze + 1/n-interp eval for snap vs fourier; write side-by-side RESULTS.md."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG))

from pspin import make_dataset, hc_torch, gs_torch  # noqa: E402
from circuit_snap import make_ansatz, fidelity, interpolate_angles  # noqa: E402
from circuit_fourier import n_params  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), "[compare-eval]", *a, flush=True)


def pack(insts, device):
    return [(hc_torch(it, device), gs_torch(it, device)) for it in insts]


@torch.no_grad()
def eval_F(sim, gamma, phi, packed) -> dict:
    Fs = [float(fidelity(sim, Hc, gs, gamma, phi).cpu()) for Hc, gs in packed]
    arr = np.asarray(Fs, dtype=np.float64)
    return {
        "median_F": float(np.median(arr)),
        "mean_F": float(arr.mean()),
        "B": int(len(arr)),
    }


def load_rows(train_dir: Path) -> list:
    p = train_dir / "train_results.json"
    if not p.exists():
        raise FileNotFoundError(p)
    rows = json.loads(p.read_text())
    # normalize: older SNAP smoke rows lack method
    for r in rows:
        if "method" not in r:
            r["method"] = "snap"
        if "n_params" not in r:
            r["n_params"] = n_params(r["method"], int(r["L"]))
        # remap ckpt paths if relative
        ck = Path(r["ckpt"])
        if not ck.is_absolute() and not ck.exists():
            alt = train_dir / ck.name
            if alt.exists():
                r["ckpt"] = str(alt)
    return rows


def best_ckpt(rows, method, Gamma, L, mode):
    cands = [
        r
        for r in rows
        if r.get("method") == method
        and float(r["Gamma"]) == float(Gamma)
        and int(r["L"]) == int(L)
        and r["mode"] == mode
    ]
    if not cands:
        return None
    return max(cands, key=lambda r: r["best_hold_score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more train dirs (each with train_results.json). Merged.",
    )
    ap.add_argument("--out-dir", type=str, default="", help="Where to write RESULTS.md")
    ap.add_argument("--eval-ns", type=int, nargs="+", default=[12, 15])
    ap.add_argument("--B-eval", type=int, default=32)
    ap.add_argument("--Gammas", type=float, nargs="+", default=[0.5])
    ap.add_argument("--Ls", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--methods", nargs="+", default=["fourier", "snap"])
    args = ap.parse_args()

    rows = []
    for d in args.train_dirs:
        rows.extend(load_rows(Path(d)))

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.train_dirs[0])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))

    eval_cache = {}
    table = []

    for method in args.methods:
        for Gamma in args.Gammas:
            for L in args.Ls:
                joint = best_ckpt(rows, method, Gamma, L, "joint")
                n6 = best_ckpt(rows, method, Gamma, L, "n6")
                n9 = best_ckpt(rows, method, Gamma, L, "n9")
                if joint is None:
                    log(f"skip {method} Γ={Gamma} L={L}: no joint ckpt")
                    continue

                st_j = torch.load(joint["ckpt"], map_location="cpu", weights_only=False)
                g_j, p_j = st_j["gamma"], st_j["phi"]

                row = {
                    "method": method,
                    "n_params": n_params(method, L),
                    "Gamma": Gamma,
                    "L": L,
                    "joint_seed": joint["seed"],
                    "joint_hold": joint["hold"],
                    "joint_train": joint["train"],
                    "freeze": {},
                    "interp": {},
                }

                for n in args.eval_ns:
                    key = (n, float(Gamma))
                    if key not in eval_cache:
                        log(f"build eval data n={n} Γ={Gamma} B={args.B_eval}")
                        insts = make_dataset(
                            n, Gamma, args.B_eval, seed0=7000 * n + int(round(10 * Gamma))
                        )
                        eval_cache[key] = pack(insts, device)
                    packed = eval_cache[key]
                    sim = make_ansatz(n, L, device)

                    fr = eval_F(sim, g_j, p_j, packed)
                    row["freeze"][str(n)] = {
                        "median_F": fr["median_F"],
                        "mean_F": fr["mean_F"],
                        "B": fr["B"],
                    }
                    log(
                        f"{method} Γ={Gamma} L={L} freeze n={n} medF={fr['median_F']:.4f}"
                    )

                    if n6 is not None and n9 is not None:
                        st6 = torch.load(n6["ckpt"], map_location="cpu", weights_only=False)
                        st9 = torch.load(n9["ckpt"], map_location="cpu", weights_only=False)
                        g_i, p_i = interpolate_angles(
                            st6["gamma"], st6["phi"], st9["gamma"], st9["phi"], n
                        )
                        ir = eval_F(sim, g_i, p_i, packed)
                        row["interp"][str(n)] = {
                            "median_F": ir["median_F"],
                            "mean_F": ir["mean_F"],
                            "B": ir["B"],
                            "n6_seed": n6["seed"],
                            "n9_seed": n9["seed"],
                        }
                        log(
                            f"{method} Γ={Gamma} L={L} interp n={n} medF={ir['median_F']:.4f}"
                        )

                table.append(row)
                (out_dir / "eval_partial.json").write_text(json.dumps(table, indent=2))

    (out_dir / "eval_results.json").write_text(json.dumps(table, indent=2))

    # Side-by-side RESULTS.md
    by_key = {(r["method"], float(r["Gamma"]), int(r["L"])): r for r in table}
    lines = [
        "# QAOA/Fourier vs Nov_3rd SNAP — quantum p-spin Γ=0.5",
        "",
        "Same cavity stack; noiseless L-BFGS; cost = 1−|⟨ψ|GS⟩|².",
        "",
        "- **Fourier (QAOA-dual):** φ_k=−β·k (k=0..3); free (β,γ)/layer → **2L params**",
        "- **SNAP:** free φ_0..3 + γ /layer → **5L params**",
        "- Train: n∈{6,9} joint (+ n6/n9-only for 1/n interp); B_train=64, B_hold=32; seeds 0–2",
        "- Transfer: hard-freeze joint; linear interp in 1/n → n=12,15 (B_eval=32)",
        "",
        "## Side-by-side (champ seed by hold MedF)",
        "",
        "| Γ | L | method | #params | hold MedF n6 | hold MedF n9 | freeze n12 | freeze n15 | interp n12 | interp n15 |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def g(d, k, field="median_F"):
        return d.get(str(k), {}).get(field, float("nan"))

    for Gamma in args.Gammas:
        for L in args.Ls:
            for method in args.methods:
                row = by_key.get((method, float(Gamma), int(L)))
                if row is None:
                    continue
                ho = row["joint_hold"]
                fr = row["freeze"]
                ir = row["interp"]
                lines.append(
                    "| {G} | {L} | {m} | {np_} | {h6:.4f} | {h9:.4f} | {f12:.4f} | {f15:.4f} | {i12:.4f} | {i15:.4f} |".format(
                        G=Gamma,
                        L=L,
                        m=method,
                        np_=row["n_params"],
                        h6=g(ho, 6),
                        h9=g(ho, 9),
                        f12=g(fr, 12),
                        f15=g(fr, 15),
                        i12=g(ir, 12),
                        i15=g(ir, 15),
                    )
                )

    # Delta table SNAP − Fourier
    lines += [
        "",
        "## Δ = SNAP − Fourier (hold / freeze)",
        "",
        "| Γ | L | Δ hold n6 | Δ hold n9 | Δ freeze n12 | Δ freeze n15 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for Gamma in args.Gammas:
        for L in args.Ls:
            s = by_key.get(("snap", float(Gamma), int(L)))
            f = by_key.get(("fourier", float(Gamma), int(L)))
            if s is None or f is None:
                continue
            lines.append(
                "| {G} | {L} | {d6:+.4f} | {d9:+.4f} | {d12:+.4f} | {d15:+.4f} |".format(
                    G=Gamma,
                    L=L,
                    d6=g(s["joint_hold"], 6) - g(f["joint_hold"], 6),
                    d9=g(s["joint_hold"], 9) - g(f["joint_hold"], 9),
                    d12=g(s["freeze"], 12) - g(f["freeze"], 12),
                    d15=g(s["freeze"], 15) - g(f["freeze"], 15),
                )
            )

    lines += ["", f"Out: `{out_dir}`", f"Train dirs: {args.train_dirs}"]
    text = "\n".join(lines) + "\n"
    (out_dir / "RESULTS.md").write_text(text)
    log(f"wrote {out_dir / 'RESULTS.md'}")
    print(text)


if __name__ == "__main__":
    main()

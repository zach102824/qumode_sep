"""Freeze + 1/n-interpolate angles → eval median F at n=12,15."""
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


def log(*a):
    print(time.strftime("%H:%M:%S"), "[qpspin-eval]", *a, flush=True)


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
        "Fs": Fs,
    }


def best_ckpt(rows, Gamma, L, mode):
    cands = [r for r in rows if r["Gamma"] == Gamma and r["L"] == L and r["mode"] == mode]
    if not cands:
        return None
    return max(cands, key=lambda r: r["best_hold_score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=str, required=True)
    ap.add_argument("--eval-ns", type=int, nargs="+", default=[12, 15])
    ap.add_argument("--B-eval", type=int, default=32)
    ap.add_argument("--Gammas", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--Ls", type=int, nargs="+", default=[1, 2, 4, 8])
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    rows = json.loads((train_dir / "train_results.json").read_text())
    device = torch.device("cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))

    eval_cache = {}
    table = []

    for Gamma in args.Gammas:
        for L in args.Ls:
            joint = best_ckpt(rows, Gamma, L, "joint")
            n6 = best_ckpt(rows, Gamma, L, "n6")
            n9 = best_ckpt(rows, Gamma, L, "n9")
            if joint is None:
                log(f"skip Γ={Gamma} L={L}: no joint ckpt")
                continue

            st_j = torch.load(joint["ckpt"], map_location="cpu", weights_only=False)
            g_j, p_j = st_j["gamma"], st_j["phi"]

            row = {
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

                # hard freeze from joint
                fr = eval_F(sim, g_j, p_j, packed)
                row["freeze"][str(n)] = {
                    "median_F": fr["median_F"],
                    "mean_F": fr["mean_F"],
                    "B": fr["B"],
                }
                log(f"Γ={Gamma} L={L} freeze n={n} medF={fr['median_F']:.4f}")

                # interpolate from n6/n9-only champs
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
                    log(f"Γ={Gamma} L={L} interp n={n} medF={ir['median_F']:.4f}")

            table.append(row)
            (train_dir / "eval_partial.json").write_text(json.dumps(table, indent=2))

    (train_dir / "eval_results.json").write_text(json.dumps(table, indent=2))

    # RESULTS.md
    lines = [
        "# quantum-p-spin smoke RESULTS",
        "",
        "Ansatz: Nov_3rd Hamming-shell SNAP (4 φ_k/cavity), cost=1−F.",
        "Train: n∈{6,9} joint (+ n6/n9-only for 1/n interp). Smoke B_train=64, B_hold=32.",
        "",
        "| Γ | L | train MedF n6 | train MedF n9 | hold MedF n6 | hold MedF n9 | freeze n12 | freeze n15 | interp n12 | interp n15 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        tr = row["joint_train"]
        ho = row["joint_hold"]
        fr = row["freeze"]
        ir = row["interp"]

        def g(d, k, field="median_F"):
            return d.get(str(k), {}).get(field, float("nan"))

        lines.append(
            "| {G} | {L} | {t6:.4f} | {t9:.4f} | {h6:.4f} | {h9:.4f} | {f12:.4f} | {f15:.4f} | {i12:.4f} | {i15:.4f} |".format(
                G=row["Gamma"],
                L=row["L"],
                t6=g(tr, 6),
                t9=g(tr, 9),
                h6=g(ho, 6),
                h9=g(ho, 9),
                f12=g(fr, 12),
                f15=g(fr, 15),
                i12=g(ir, 12),
                i15=g(ir, 15),
            )
        )
    lines.append("")
    lines.append(f"Train dir: `{train_dir}`")
    (train_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {train_dir / 'RESULTS.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Publish exact L5/L6 records from round5 cells. No new training."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round5_S4"))
import sim

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "round5_S4"
CELLS = SRC / "results" / "cells"
STRIDE4 = [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(name: str):
    path = CELLS / f"{name}.json"
    cell = json.loads(path.read_text())
    start = min(cell["starts"], key=lambda s: s["energy_error"])
    return path, start


def main():
    if sha(SRC / "FROZEN_TARGET_R2.json") != sim.EXPECTED:
        raise RuntimeError("manifest")
    sysd = sim.parse_system(STRIDE4)
    spec = sim.spectrum(sysd)
    out = {
        "recovered_from_existing_cells": True,
        "source_commit": "d9af7f2a",
        "manifest_sha256": sim.EXPECTED,
        "packing_sha256": sha(SRC / "stride4_packing.json"),
        "E0": spec["E0"],
        "full_gap": spec["full_gap"],
        "states": {},
    }
    psis = {}
    for label, name, L in (
        ("L5_energy_selected", "arm0_L5", 5),
        ("L6_energy_selected", "arm2_L6", 6),
        ("L6_arm0_same_ray", "arm0_L6", 6),
        ("L6_arm1_nearby", "arm1_L6", 6),
    ):
        path, start = record(name)
        x = np.asarray(start["x"], dtype=np.float64)
        psi = sim.logical_state(sysd, L, x)
        E, var = sim.energy_var(psi, sysd)
        F = sim.fidelity(spec["gs"], psi)
        digest = hashlib.sha256(np.ascontiguousarray(psi).tobytes()).hexdigest()
        g, tk, tm, shells = sim.unpack(x, L)
        parity = float(np.vdot(psi, psi[::-1]).real)
        out["states"][label] = {
            "cell": f"Nov_3rd/tfim_shared_gamma/round5_S4/results/cells/{name}.json",
            "cell_sha256": sha(path),
            "start_k": start["k"],
            "role": start["role"],
            "energy_error": float(E - spec["E0"]),
            "stored_energy_error": start["energy_error"],
            "F": F,
            "stored_F": start["F"],
            "variance": var,
            "stored_variance": start["var_H"],
            "norm": float(np.linalg.norm(psi)),
            "parity_expectation": parity,
            "state_hash": digest,
            "stored_state_hash": start["state_hash"],
            "hash_matches_stored": digest == start["state_hash"],
            "x": x.tolist(),
            "angles": {
                "gamma": g,
                "theta_k": tk,
                "theta_m": tm,
                "c1": shells[:, 0].tolist(),
                "c2": shells[:, 1].tolist(),
                "c3": shells[:, 2].tolist(),
            },
        }
        psis[label] = psi
        print(label, "E", E - spec["E0"], "F", F, "var", var, "hash", digest[:16], "match", digest == start["state_hash"])
    out["overlaps"] = {
        "L6_arm0_vs_selected": abs(np.vdot(psis["L6_arm0_same_ray"], psis["L6_energy_selected"])),
        "L6_arm1_vs_selected": abs(np.vdot(psis["L6_arm1_nearby"], psis["L6_energy_selected"])),
        "L6_arm1_vs_selected_conjugate": abs(np.vdot(np.conj(psis["L6_arm1_nearby"]), psis["L6_energy_selected"])),
    }
    dest = ROOT / "published_L5_L6.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    np.savez_compressed(ROOT / "published_states.npz", L5=psis["L5_energy_selected"], L6=psis["L6_energy_selected"])
    print("wrote", dest, "sha", sha(dest))


if __name__ == "__main__":
    main()

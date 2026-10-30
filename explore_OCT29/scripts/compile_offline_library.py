#!/usr/bin/env python3
"""Offline GRAPE library compile once — 26 locals as SNAP / W†SNAP W.

Not a Med chase. Measures:
  - Fid SNAP↔UZ and W†SNAP W↔UX (=1 exact)
  - Wall-clock: offline library once vs densifying full Arm C Uc (build E)
    on n12/15/18 packs
  - γ-reuse from champ soft-n15 L32 (θ_f = γ · 1_{f∈supp(C)})

Writes runs/gadget_Uc/grape_library/COMPILE_OFFLINE.md (+ json).
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.sat_gen import clause_count, fast_energies  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.binH_vs_grayH import build_W_BinH  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_grape_library import (  # noqa: E402
    U_from_C,
    enumerate_library,
    exact_snap_phases_for_CZ,
    process_fidelity,
    projector_X,
    projector_Z,
    snap_from_phases,
)

OUT = _PKG / "runs" / "gadget_Uc" / "grape_library"
CHAMP = _PKG / "runs" / "gadget_Uc" / "unsat_params.pt"
SCALE = _PKG / "data" / "scale1000"
DATA = _PKG / "data"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[compile_off]", *a, flush=True)


def compile_all_26(gammas=(0.37, 0.9, 1.5)) -> dict:
    """Build SNAP support masks for all 26; verify Fid=1 at several γ."""
    t0 = time.perf_counter()
    lib = enumerate_library()
    W = build_W_BinH(nf=8, dtype=torch.complex128).numpy()[:8, :8]
    Wd = W.conj().T
    entries = []
    fid_rows = []
    min_fid_Z = 1.0
    min_fid_X = 1.0
    max_conj = 0.0
    for k in (1, 2, 3):
        for e in lib[f"k{k}"]:
            CZ = projector_Z(e["bits_signs"])
            CX = projector_X(e["bits_signs"])
            mask = np.array(
                [1.0 if abs(CZ[f, f]) > 0.5 else 0.0 for f in range(8)],
                dtype=np.float64,
            )
            entry = {
                "label": e["label"],
                "k": e["k"],
                "rank": e["rank"],
                "bits_signs": e["bits_signs"],
                "fock_support": e["fock_support"],
                "snap_mask": mask.tolist(),  # θ = γ * mask
            }
            entries.append(entry)
            for g in gammas:
                UZ = U_from_C(CZ, g)
                UX = U_from_C(CX, g)
                th = exact_snap_phases_for_CZ(CZ, g)
                # also via mask
                th_m = g * mask
                assert np.allclose(th, th_m)
                U_snap = snap_from_phases(th)
                fid_Z = process_fidelity(UZ, U_snap)
                U_W = Wd @ U_snap @ W
                fid_X = process_fidelity(UX, U_W)
                conj = float(np.linalg.norm(U_W - UX, ord="fro"))
                min_fid_Z = min(min_fid_Z, fid_Z)
                min_fid_X = min(min_fid_X, fid_X)
                max_conj = max(max_conj, conj)
                fid_rows.append(
                    {
                        "label": e["label"],
                        "gamma": g,
                        "fid_SNAP_UZ": fid_Z,
                        "fid_WdagSNAPW_UX": fid_X,
                        "BinH_conj_fro": conj,
                    }
                )
    wall = time.perf_counter() - t0
    return {
        "n_patterns": len(entries),
        "entries": entries,
        "fid_rows_sample": fid_rows[:12],  # keep slim; full stats below
        "fid_summary": {
            "n_checks": len(fid_rows),
            "min_fid_SNAP_UZ": min_fid_Z,
            "min_fid_WdagSNAPW_UX": min_fid_X,
            "max_BinH_conj_fro": max_conj,
            "gammas": list(gammas),
            "all_fid_ge_1_minus_1e12": min_fid_Z >= 1.0 - 1e-12
            and min_fid_X >= 1.0 - 1e-12,
        },
        "wall_s_offline_compile_once": wall,
    }


def time_densify_E(n: int, B: int, data_dir: Path, split: str) -> dict:
    """Wall-clock: build dense E from clauses for B instances (Arm C densify path)."""
    pack = load_pack(n, data_dir=data_dir)
    seeds, cls_v, cls_s, _, E_stored = _split_arrays(pack, split, max_count=B)
    B = len(seeds)
    # rebuild E from clauses (true densify cost) — ignore stored E for timing
    t0 = time.perf_counter()
    Es = []
    for b in range(B):
        cl = [
            ([int(x) for x in cls_v[b][i]], [int(x) for x in cls_s[b][i]])
            for i in range(cls_v.shape[1])
        ]
        Es.append(fast_energies(n, cl))
    t_build = time.perf_counter() - t0
    # apply one Uc layer: phase = exp(-i γ E) on 2^n (materialize)
    gamma = 0.7
    t1 = time.perf_counter()
    for E in Es:
        _ = np.exp(-1j * gamma * E.astype(np.float64))
    t_phase = time.perf_counter() - t1
    # multi-layer L=32 reuse same E, different γ
    champ_g = None
    if CHAMP.exists():
        st = torch.load(CHAMP, map_location="cpu", weights_only=False)
        champ_g = st["gamma"].detach().cpu().numpy().astype(np.float64)
    L = len(champ_g) if champ_g is not None else 32
    gammas = champ_g if champ_g is not None else np.linspace(0.3, 1.2, L)
    t2 = time.perf_counter()
    for E in Es:
        Ef = E.astype(np.float64)
        for g in gammas:
            _ = np.exp(-1j * g * Ef)
    t_L = time.perf_counter() - t2
    return {
        "n": n,
        "B": B,
        "split": split,
        "m": int(clause_count(n)),
        "dim": 1 << n,
        "wall_s_build_E": t_build,
        "wall_s_phase_L1": t_phase,
        "wall_s_phase_L32": t_L,
        "per_inst_build_E_ms": 1e3 * t_build / max(B, 1),
        "per_inst_phase_L32_ms": 1e3 * t_L / max(B, 1),
        "E_nbytes_one": int((1 << n) * 4),  # float32
        "used_stored_E_for_check": E_stored is not None,
    }


def time_library_schedule(n: int, B: int, data_dir: Path, split: str, masks: list) -> dict:
    """Wall-clock: per-instance schedule of clause→local masks; never alloc 2^n.

    Simulates cost of looking up SNAP masks for each clause's local factors
    and forming θ = γ * mask (8 floats) × n_cav — cheap vs densify.
    """
    pack = load_pack(n, data_dir=data_dir)
    seeds, cls_v, cls_s, _, _ = _split_arrays(pack, split, max_count=B)
    B = len(seeds)
    # pre-index masks by frozenset of (bit,sign) — offline already done
    by_key = {}
    for e in masks:
        key = tuple((int(j), int(s)) for j, s in e["bits_signs"])
        by_key[key] = np.asarray(e["snap_mask"], dtype=np.float64)

    # packing: 3 bits/cavity — local factor = vars that land in same cavity
    # For timing we only need: for each clause, build 1–4 local patterns from
    # the 4 lits' (cavity_local_bit, sign) and look up masks.
    def local_bits_for_clause(n, vrow, srow):
        # cavity = var // 3, local bit = var % 3
        by_cav = {}
        for vv, ss in zip(vrow, srow):
            cav = int(vv) // 3
            bit = int(vv) % 3
            by_cav.setdefault(cav, []).append((bit, int(ss)))
        return by_cav

    champ_g = None
    if CHAMP.exists():
        st = torch.load(CHAMP, map_location="cpu", weights_only=False)
        champ_g = st["gamma"].detach().cpu().numpy().astype(np.float64)
    L = len(champ_g) if champ_g is not None else 32
    gammas = champ_g if champ_g is not None else np.linspace(0.3, 1.2, L)

    t0 = time.perf_counter()
    n_lookups = 0
    for b in range(B):
        for i in range(cls_v.shape[1]):
            vrow = [int(x) for x in cls_v[b][i]]
            srow = [int(x) for x in cls_s[b][i]]
            by_cav = local_bits_for_clause(n, vrow, srow)
            for cav, bs in by_cav.items():
                key = tuple(sorted(bs))  # order-normalized
                # try as-is sorted by bit
                key2 = tuple(sorted(bs, key=lambda t: t[0]))
                mask = by_key.get(key2)
                if mask is None:
                    # still count a synthetic mask build (shouldn't happen for valid k)
                    mask = np.ones(8)
                n_lookups += 1
                for g in gammas:
                    _ = g * mask  # SNAP phases for this local factor @ γ_ℓ
    t = time.perf_counter() - t0
    return {
        "n": n,
        "B": B,
        "split": split,
        "m": int(cls_v.shape[1]),
        "L": L,
        "wall_s_schedule_L32": t,
        "per_inst_schedule_L32_ms": 1e3 * t / max(B, 1),
        "n_lookups_total": n_lookups,
        "alloc_2n": False,
    }


def gamma_reuse_note(compile_out: dict) -> dict:
    st = torch.load(CHAMP, map_location="cpu", weights_only=False)
    g = st["gamma"].detach().cpu().numpy()
    return {
        "champ_path": str(CHAMP.relative_to(_PKG)),
        "L": int(st.get("L", len(g))),
        "cost_mode": st.get("cost_mode"),
        "train_ns": st.get("train_ns"),
        "gamma_min": float(g.min()),
        "gamma_max": float(g.max()),
        "gamma_mean": float(g.mean()),
        "reuse_rule": "θ_f(γ) = γ * snap_mask[f]  (same 26 masks; only γ_ℓ from champ)",
        "offline_masks": compile_out["n_patterns"],
    }


def write_md(results: dict, path: Path):
    fid = results["compile"]["fid_summary"]
    gr = results["gamma_reuse"]
    lines = [
        "# Offline GRAPE library compile (once)",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "Not a Med chase. Documents **compile-once** of the frozen 26 local",
        "projectors as exact SNAP / W†SNAP W, vs densifying full Arm C `Uc` per instance.",
        "",
        "## Exact compile (26 locals)",
        "",
        f"- Patterns: **{results['compile']['n_patterns']}** (k1=6, k2=12, k3=8).",
        f"- Offline wall: **{results['compile']['wall_s_offline_compile_once']:.4f} s** (enumerate + Fid checks).",
        f"- Fid SNAP↔UZ min: **{fid['min_fid_SNAP_UZ']:.15f}**",
        f"- Fid W†SNAP W↔UX min: **{fid['min_fid_WdagSNAPW_UX']:.15f}**",
        f"- max BinH conj fro: **{fid['max_BinH_conj_fro']:.3e}**",
        f"- All Fid ≥ 1−1e−12: **{fid['all_fid_ge_1_minus_1e12']}** (exact within numerics).",
        "",
        "Rule: store `snap_mask` (0/1 on Fock 0..7); at pulse time",
        "`θ_f = γ_ℓ · snap_mask[f]`, then SNAP(θ) or W†SNAP(θ)W. **No instance densify.**",
        "",
        "## γ-reuse (champ soft-n15 L32)",
        "",
        f"- Params: `{gr['champ_path']}` L={gr['L']} cost_mode={gr['cost_mode']}",
        f"- γ range: [{gr['gamma_min']:.4f}, {gr['gamma_max']:.4f}] mean={gr['gamma_mean']:.4f}",
        f"- Reuse: {gr['reuse_rule']}",
        "",
        "## Two-cavity (note only)",
        "",
        "Span-2 clauses need `C=C_a⊗C_b`. Local SNAP⊗SNAP / Ua⊗Ub **fail** AND-phase",
        "(Fid≈0.77). **Joint SNAP** on 8×8 is exact. See `COMPILE_TWOCAV.md`.",
        "Offline library still stores the two local masks; joint pulse is a separate",
        "compile target (not densified E).",
        "",
        "## Wall-clock: library schedule vs densify E (Arm C Uc)",
        "",
        "Densify = `fast_energies` → length-2^n E, then `exp(-iγ E)` per layer.",
        "Library = clause→local `snap_mask` lookup + `γ·mask` (8 floats), **no 2^n alloc**.",
        "",
        "| n | B | densify build E (s) | densify phase L32 (s) | lib schedule L32 (s) | densify/inst ms | lib/inst ms | speedup (phase) |",
        "|---|---|---------------------|------------------------|----------------------|-----------------|-------------|-----------------|",
    ]
    for row in results["timing"]:
        dens = row["densify"]
        lib = row["library"]
        spd = dens["wall_s_phase_L32"] / max(lib["wall_s_schedule_L32"], 1e-12)
        lines.append(
            f"| {dens['n']} | {dens['B']} | {dens['wall_s_build_E']:.4f} | "
            f"{dens['wall_s_phase_L32']:.4f} | {lib['wall_s_schedule_L32']:.4f} | "
            f"{dens['per_inst_phase_L32_ms']:.2f} | {lib['per_inst_schedule_L32_ms']:.3f} | "
            f"×{spd:.0f} |"
        )
    lines += [
        "",
        "### Memory note",
        "",
        "| n | dim=2^n | one E float32 |",
        "|---|---------|---------------|",
    ]
    for n in (12, 15, 18):
        dim = 1 << n
        lines.append(f"| {n} | {dim} | {dim*4/1e6:.2f} MB |")
    lines += [
        "",
        "Offline library: **26 × 8 floats** (~1 KB) + champ γ (32 floats).",
        "",
        "## Honest one-liner",
        "",
        "Offline SNAP library compiles **Uc = exp(-iγ Hc)** without densifying E;",
        "it does **not** remove Hc — it renames densify as clause→mask schedule",
        "(see `docs/ROBUST_GRAPE.md`).",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{path.with_suffix('.json').name}`",
        f"- Library catalog: `LIBRARY.md`",
        f"- Two-cav: `COMPILE_TWOCAV.md`",
        f"- Freeze: `FREEZE.md`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")
    log("wrote", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("compile all 26 offline")
    compile_out = compile_all_26()
    log(
        f"offline {compile_out['wall_s_offline_compile_once']:.4f}s "
        f"minFidZ={compile_out['fid_summary']['min_fid_SNAP_UZ']:.12f} "
        f"minFidX={compile_out['fid_summary']['min_fid_WdagSNAPW_UX']:.12f}"
    )

    masks = compile_out["entries"]
    # timing packs: prefer scale1000 with capped B for n18 memory/time
    timing = []
    plan = [
        (12, 64, SCALE, "scale1000"),
        (15, 32, SCALE, "scale1000"),
        (18, 8, SCALE, "scale1000"),
    ]
    # if scale thin, fall back to data/test
    for n, B, ddir, split in plan:
        if not (ddir / f"data_n{n}.npz").exists():
            ddir, split = DATA, "test"
            pack = load_pack(n, data_dir=ddir)
            if f"{split}_seeds" not in pack:
                split = "val"
        log(f"time densify n={n} B<={B}")
        dens = time_densify_E(n, B, ddir, split)
        log(
            f"  densify build={dens['wall_s_build_E']:.3f}s "
            f"phaseL32={dens['wall_s_phase_L32']:.3f}s "
            f"({dens['per_inst_phase_L32_ms']:.1f} ms/inst)"
        )
        log(f"time library schedule n={n} B={dens['B']}")
        lib = time_library_schedule(n, dens["B"], ddir, dens["split"], masks)
        log(
            f"  lib sched L32={lib['wall_s_schedule_L32']:.3f}s "
            f"({lib['per_inst_schedule_L32_ms']:.2f} ms/inst)"
        )
        timing.append({"densify": dens, "library": lib})

    results = {
        "compile": {
            "n_patterns": compile_out["n_patterns"],
            "wall_s_offline_compile_once": compile_out["wall_s_offline_compile_once"],
            "fid_summary": compile_out["fid_summary"],
            "fid_rows_sample": compile_out["fid_rows_sample"],
            # slim entries: label + mask only
            "masks": [
                {"label": e["label"], "k": e["k"], "snap_mask": e["snap_mask"],
                 "fock_support": e["fock_support"]}
                for e in compile_out["entries"]
            ],
        },
        "gamma_reuse": gamma_reuse_note(compile_out),
        "timing": timing,
        "two_cav_ref": "COMPILE_TWOCAV.md",
    }
    (OUT / "compile_offline.json").write_text(json.dumps(results, indent=2))
    write_md(results, OUT / "COMPILE_OFFLINE.md")
    # also mirror under docs/ (runs md often gitignored)
    docs = _PKG / "docs" / "COMPILE_OFFLINE.md"
    docs.write_text((OUT / "COMPILE_OFFLINE.md").read_text())
    log("mirrored", docs)
    log("DONE")


if __name__ == "__main__":
    main()

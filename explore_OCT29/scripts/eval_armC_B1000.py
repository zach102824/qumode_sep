#!/usr/bin/env python3
"""Eval probe6 Arm C on data/scale1000 packs (B=1000) and write RESULTS.

Uses saved shell params; zero-shot transfer to n15/n18. Incremental JSON after
each n. Fits b_Med / b_B90 on 9→18 and pairwise windows.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.ansatz import AnsatzConfig  # noqa: E402
from explore_OCT29.lib.drivers import fit_b  # noqa: E402
from explore_OCT29.lib.spectral_arm import build_W_H, load_arm_C  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.probe6_spectral_mixer import SpectralAnsatz  # noqa: E402

OUT = _PKG / "runs" / "armC_B1000"
DATA = _PKG / "data" / "scale1000"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[armC_B1000]", *a, flush=True)


def _S(med, b90):
    return 0.5 * (math.log(max(med, 1e-12)) + math.log(max(b90, 1e-12)))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if o is None:
        return None
    return o


@torch.no_grad()
def eval_n(n, W, gamma, phi, L, nf, device, data_dir, split="scale1000",
           max_count=1000, progress_every=25):
    pack = load_pack(n, data_dir=data_dir)
    # Prefer scale1000_* keys; fall back to test_*
    key = split
    if f"{split}_seeds" not in pack:
        key = "test"
    seeds, _, _, _, E = _split_arrays(pack, key, max_count=max_count)
    if E is None or len(seeds) == 0:
        raise RuntimeError(f"n={n}: no dense E for split={key} in {data_dir}")
    cfg = AnsatzConfig(n=n, L=L, nf=nf)
    sim = SpectralAnsatz(cfg, W, device=device).to(device)
    invs, leaks = [], []
    t0 = time.time()
    B = len(seeds)
    for b in range(B):
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        out = sim.forward_spectral(Eb, gamma, "shell", phi=phi)
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
        if (b + 1) % progress_every == 0 or b + 1 == B:
            elapsed = time.time() - t0
            rate = (b + 1) / max(elapsed, 1e-9)
            eta = (B - b - 1) / max(rate, 1e-9)
            med_so_far = float(np.median(invs))
            log(
                f"n={n} {b+1}/{B} Med≈{med_so_far:.3f} "
                f"rate={rate:.2f}/s ETA={eta/60:.1f}min"
            )
    invs_a = np.asarray(invs, dtype=np.float64)
    leaks_a = np.asarray(leaks, dtype=np.float64)
    med = float(np.median(invs_a))
    b90 = float(np.quantile(invs_a, 0.9))
    return {
        "n": n,
        "split": key,
        "B": int(B),
        "median_inv_p": med,
        "B90_inv_p": b90,
        "S": _S(med, b90),
        "mean_leak": float(np.mean(leaks_a)),
        "frac_inv_le_2": float(np.mean(invs_a <= 2.0)),
        "mean_inv_p": float(np.mean(invs_a)),
        "wall_s": time.time() - t0,
        "L": L,
        "L_times_Med": L * med,
        "mode": "shell",
        "inv_p_all": invs_a.tolist(),  # keep for bootstrap later; large but ok
    }


def fit_windows(ladder: dict):
    windows = {
        "9_18": (9, 12, 15, 18),
        "9_15": (9, 12, 15),
        "12_18": (12, 15, 18),
        "9_12": (9, 12),
        "12_15": (12, 15),
        "15_18": (15, 18),
    }
    out = {}
    for name, ns in windows.items():
        ns_u, meds, b90s = [], [], []
        for n in ns:
            m = ladder.get(n) or ladder.get(str(n))
            if m:
                ns_u.append(int(n))
                meds.append(m["median_inv_p"])
                b90s.append(m["B90_inv_p"])
        if len(ns_u) < 2:
            continue
        out[name] = {
            "ns": ns_u,
            "b_Med": fit_b(ns_u, meds),
            "b_B90": fit_b(ns_u, b90s),
        }
    return out


def write_results_md(path: Path, meta: dict, ladder: dict, fits: dict):
    lines = [
        "# Arm C — B=1000 scaling eval",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        f"**Motif:** probe6 Arm C free shells — prep=`W_H`, layer `COST → W_H·SNAP(φ_pc)·W_H†`, L={meta['L']}",
        f"**Params:** `{meta['params']}`",
        f"**Pack:** `{meta['data_dir']}` split=`{meta['split']}` seed base SCB=54e9",
        f"**B:** {meta['B']} per n (requested)",
        "",
        "## Ladder (scale1000)",
        "",
        "| n | Med | B90 | leak | L×Med | frac_inv≤2 | B | wall_min |",
        "|---|-----|-----|------|-------|------------|---|----------|",
    ]
    for n in sorted(int(k) for k in ladder.keys()):
        m = ladder[n] if n in ladder else ladder[str(n)]
        lines.append(
            f"| {n} | {m['median_inv_p']:.3f} | {m['B90_inv_p']:.3f} | "
            f"{m['mean_leak']:.4f} | {m['L_times_Med']:.1f} | "
            f"{m['frac_inv_le_2']:.3f} | {m['B']} | {m['wall_s']/60:.1f} |"
        )
    lines += ["", "## b fits", ""]
    for name, f in fits.items():
        bm = f["b_Med"]["b"]
        bb = f["b_B90"]["b"]
        lines.append(
            f"- **{name}** (ns={f['ns']}): b_Med≈**{bm:.4f}**, b_B90≈**{bb:.4f}**"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Sealed small test packs under `explore_OCT29/data/data_n*.npz` untouched.",
        "- Prior small-B test (probe6): n15 Med≈4.50 B=40; n18 Med≈3.76 B=16 — B=1000 is the honest ladder.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", type=str, default="9,12,15,18")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--nf", type=int, default=16)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--data_dir", type=str, default=str(DATA))
    ap.add_argument("--params", type=str, default=None)
    ap.add_argument("--split", type=str, default="scale1000")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--progress_every", type=int, default=25)
    ap.add_argument("--skip_inv_all", action="store_true",
                    help="drop per-instance inv_p from JSON to keep it small")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)

    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    arm = load_arm_C(Path(args.params) if args.params else None)
    L = int(args.L) if args.L else arm["L"]
    gamma = arm["gamma"]
    phi = arm["phi"]
    if gamma.numel() != L or phi.shape[0] != L:
        raise SystemExit(f"L mismatch: args/arm L={L} gamma={gamma.numel()} phi={phi.shape}")

    device = torch.device(args.device)
    W = build_W_H(args.nf)
    ns = [int(x) for x in args.ns.split(",") if x.strip()]

    meta = {
        "L": L,
        "nf": args.nf,
        "B": args.B,
        "params": arm["path"],
        "data_dir": str(data_dir),
        "split": args.split,
        "device": str(device),
        "threads": args.threads,
        "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    ladder = {}
    results_path = OUT / "results.json"
    # resume if partial
    if results_path.exists():
        try:
            prev = json.loads(results_path.read_text())
            for k, v in (prev.get("ladder") or {}).items():
                ladder[int(k)] = v
            log(f"resumed partial ladder ns={sorted(ladder)}")
        except Exception as e:
            log("resume skip", e)

    for n in ns:
        if n in ladder and ladder[n].get("B", 0) >= args.B:
            log(f"skip n={n} already done B={ladder[n]['B']}")
            continue
        log(f"=== eval n={n} B<={args.B} ===")
        m = eval_n(
            n, W, gamma, phi, L, args.nf, device, data_dir,
            split=args.split, max_count=args.B,
            progress_every=args.progress_every,
        )
        if args.skip_inv_all:
            m = {k: v for k, v in m.items() if k != "inv_p_all"}
        ladder[n] = m
        log(
            f"DONE n={n}: Med={m['median_inv_p']:.3f} B90={m['B90_inv_p']:.3f} "
            f"leak={m['mean_leak']:.4f} L×Med={m['L_times_Med']:.1f} "
            f"frac≤2={m['frac_inv_le_2']:.3f} B={m['B']} wall={m['wall_s']/60:.1f}min"
        )
        fits = fit_windows(ladder)
        blob = {
            "meta": meta,
            "ladder": {str(k): v for k, v in sorted(ladder.items())},
            "fits": _jsonable(fits),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        }
        results_path.write_text(json.dumps(_jsonable(blob), indent=2))
        write_results_md(OUT / "RESULTS.md", meta, ladder, fits)
        log(f"wrote {results_path}")

    fits = fit_windows(ladder)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    blob = {
        "meta": meta,
        "ladder": {str(k): v for k, v in sorted(ladder.items())},
        "fits": _jsonable(fits),
        "updated": meta["finished"],
    }
    # drop inv_p_all from final if huge — keep sidecar
    slim = _jsonable(blob)
    for n_s, m in slim["ladder"].items():
        invs = m.pop("inv_p_all", None)
        if invs is not None:
            np.save(OUT / f"inv_p_n{n_s}.npy", np.asarray(invs, dtype=np.float64))
    results_path.write_text(json.dumps(slim, indent=2))
    write_results_md(OUT / "RESULTS.md", meta, ladder, fits)
    log("ALL DONE")
    for name, f in fits.items():
        log(f"fit {name}: b_Med={f['b_Med']['b']:.4f} b_B90={f['b_B90']['b']:.4f}")


if __name__ == "__main__":
    main()

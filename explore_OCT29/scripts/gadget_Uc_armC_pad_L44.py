#!/usr/bin/env python3
"""Fair ArmC pad L16→L44 on scale1000 (NON-DEPTH wave #1).

Interp Arm C shell angles to L=44 (same pad used for projector depth ladder).
Eval SpectralAnsatz densified-E. Honesty: ArmC still densified Hc; fair L match only.
Writes runs/gadget_Uc/renames_Hc/armC_L44_pad/
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
from explore_OCT29.scripts.gadget_Uc_depth import interp_angles  # noqa: E402
from explore_OCT29.scripts.probe6_spectral_mixer import SpectralAnsatz  # noqa: E402

OUT = _PKG / "runs" / "gadget_Uc" / "renames_Hc" / "armC_L44_pad"
DATA = _PKG / "data" / "scale1000"
ARMC = _PKG / "runs" / "probe6_spectral_mixer" / "arm_C_shell_params.pt"
CHAMP_REF = {
    9: (1.051, 1.473),
    12: (1.113, 2.263),
    15: (1.233, 4.243),
    18: (1.409, 9.645),
}
ARMC_L16_REF = {
    9: (1.441, 3.314),
    12: (1.991, 6.252),
    15: (2.921, 10.959),
    18: (4.573, 27.810),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[armC_L44]", *a, flush=True)


def _S(med, b90):
    return 0.5 * (math.log(max(med, 1e-12)) + math.log(max(b90, 1e-12)))


@torch.no_grad()
def eval_n(n, W, gamma, phi, L, nf, device, data_dir, split="scale1000",
           max_count=1000, progress_every=50):
    pack = load_pack(n, data_dir=data_dir)
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
            log(
                f"n={n} {b+1}/{B} Med≈{float(np.median(invs)):.3f} "
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
        "mode": "shell_pad",
    }


def fit_windows(ladder: dict):
    windows = {
        "9_18": (9, 12, 15, 18),
        "9_15": (9, 12, 15),
        "12_18": (12, 15, 18),
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


def write_md(path: Path, meta: dict, ladder: dict, fits: dict):
    lines = [
        "# Fair ArmC pad L16→L44 — scale1000",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "Honesty: ArmC = densified `exp(-iγ Hc)` in W_H frame; pad = linear interp of shell "
        "angles only (same recipe as projector depth pad). **Fair same-L baseline**, not a promote.",
        "",
        f"**L:** {meta['L']} (from {meta['L0']})  **B:** {meta['B']}  **nf:** {meta['nf']}",
        f"**Params:** `{meta['params']}` → `{meta['out_params']}`",
        "",
        "## Ladder",
        "",
        "| n | ArmC L44 Med/B90 | ArmC L16 | champ L44 | L×Med | wall_min |",
        "|---|------------------|----------|-----------|-------|----------|",
    ]
    for n in (9, 12, 15, 18):
        m = ladder.get(n) or ladder.get(str(n))
        if not m:
            continue
        a16 = ARMC_L16_REF[n]
        ch = CHAMP_REF[n]
        lines.append(
            f"| {n} | **{m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}** | "
            f"{a16[0]:.3f}/{a16[1]:.3f} | {ch[0]:.3f}/{ch[1]:.3f} | "
            f"{m['L_times_Med']:.1f} | {m['wall_s']/60:.1f} |"
        )
    lines += ["", "## b fits", ""]
    for name, f in fits.items():
        lines.append(
            f"- **{name}**: b_Med≈**{f['b_Med']['b']:.4f}**, b_B90≈**{f['b_B90']['b']:.4f}**"
        )
    # vs refs
    lines += ["", "## vs ArmC L16 (pad hurt?)", ""]
    hurt = []
    for n in (9, 12, 15, 18):
        m = ladder.get(n) or ladder.get(str(n))
        if not m:
            continue
        ratio = m["median_inv_p"] / ARMC_L16_REF[n][0]
        hurt.append(ratio >= 1.0)
        lines.append(
            f"- n{n}: Med ratio L44/L16 = {ratio:.3f} "
            f"({'hurt or flat' if ratio >= 1.0 else 'improved'})"
        )
    if hurt and all(hurt):
        lines.append(
            "\n**Note:** pad hurts/flat on every n — still publish as negative fair baseline."
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--L", type=int, default=44)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--nf", type=int, default=16)
    ap.add_argument("--ns", type=str, default="9,12,15,18")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--progress_every", type=int, default=50)
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)

    OUT.mkdir(parents=True, exist_ok=True)
    arm = load_arm_C(ARMC)
    g0, p0 = arm["gamma"], arm["phi"]
    L0 = int(g0.numel())
    g, p = interp_angles(g0, p0, args.L)
    out_pt = OUT / f"arm_C_shell_params_L{args.L}_pad.pt"
    torch.save(
        {
            "gamma": g,
            "phi": p,
            "mode": "shell",
            "L": args.L,
            "pad_from": L0,
            "source": str(ARMC),
        },
        out_pt,
    )
    log(f"saved pad params L{L0}→{args.L} → {out_pt}")

    device = torch.device(args.device)
    W = build_W_H(args.nf)
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    meta = {
        "L": args.L,
        "L0": L0,
        "nf": args.nf,
        "B": args.B,
        "params": str(ARMC),
        "out_params": str(out_pt),
        "data_dir": str(DATA),
        "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    ladder = {}
    results_path = OUT / "results.json"
    if results_path.exists():
        try:
            prev = json.loads(results_path.read_text())
            for k, v in (prev.get("ladder") or {}).items():
                if int(v.get("B", 0)) >= args.B and int(v.get("L", 0)) == args.L:
                    ladder[int(k)] = v
            log(f"resumed ns={sorted(ladder)}")
        except Exception as e:
            log("resume skip", e)

    for n in ns:
        if n in ladder:
            log(f"skip n={n}")
            continue
        log(f"=== eval n={n} L={args.L} B<={args.B} ===")
        m = eval_n(
            n, W, g, p, args.L, args.nf, device, DATA,
            max_count=args.B, progress_every=args.progress_every,
        )
        ladder[n] = m
        payload = {
            "meta": meta,
            "ladder": {str(k): v for k, v in ladder.items()},
            "fits": {},
            "champ_L44_ref": CHAMP_REF,
            "armC_L16_ref": ARMC_L16_REF,
            "honesty": "ArmC densified Hc; fair L44 pad baseline",
        }
        # strip nothing heavy
        results_path.write_text(json.dumps(payload, indent=2))
        log(f"n{n} Med/B90={m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}")

    fits = fit_windows(ladder)
    payload = {
        "meta": {**meta, "finished": time.strftime("%Y-%m-%d %H:%M:%S %z")},
        "ladder": {str(k): v for k, v in ladder.items()},
        "fits": {
            k: {
                "ns": v["ns"],
                "b_Med": float(v["b_Med"]["b"]),
                "b_B90": float(v["b_B90"]["b"]),
            }
            for k, v in fits.items()
        },
        "champ_L44_ref": {str(k): list(v) for k, v in CHAMP_REF.items()},
        "armC_L16_ref": {str(k): list(v) for k, v in ARMC_L16_REF.items()},
        "honesty": "ArmC densified Hc; fair L44 pad baseline",
    }
    results_path.write_text(json.dumps(payload, indent=2))
    write_md(OUT / "RESULTS.md", meta, ladder, fits)
    log("DONE", OUT / "RESULTS.md")


if __name__ == "__main__":
    main()

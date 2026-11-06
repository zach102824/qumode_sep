#!/usr/bin/env python3
"""Retrain frozen-champ families on π_max-packed train{6,9}.

Only run after the CAP smoke PASSes. Train packs are the same instances
the champs used, with f'(x) = f(π_max^{-1} x) stored as train_E.
Eval Med is success on the original formula under that relabeling.

Comparisons:
  L=2 Walsh, 3 seeds, vs Walsh seed1 knee R≈0.9567 (sealed QAOA denominator)
  L=16 power_p, 3 seeds, vs fresh250 power_p and Fourier QAOA Q=6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCR = Path(__file__).resolve().parent
if str(_SCR) not in sys.path:
    sys.path.insert(0, str(_SCR))

from beat_params import phi_from_c
from cap_smoke import EVAL_NS, PUBLISHED, eval_snap, load_core, log, strategies_for_n
from Nov_3rd import settings as S
from Nov_3rd.cavity_packing import pack_batch, permute_energy_instances
from explore_OCT29.lib.sat_gen import gen_instance, pack_clauses

PY = sys.executable
SEEDS = (0, 1, 2)
WALSH_SRC = S.OUT / "beat_qaoa_12h" / "train_packs"
POWER_SRC = S.OUT / "beat_qaoa_12h" / "train69_ladder_fresh250" / "train_packs"
POWER_QAOA = S.OUT / "beat_qaoa_12h" / "train69_ladder_fresh250" / "qaoa_angles"
# Matched Fourier QAOA Q=6 L=16 on this CORE (FM-WS branch eval).
FOURIER_L16 = {6: 1.1471417505862727, 9: 1.4111617074249911, 12: 1.7653476890915212, 15: 2.3853191788935257}


def build_packed(src: Path, dst: Path, ns=(6, 9)) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for n in ns:
        z = np.load(src / f"train_n{n}.npz", allow_pickle=True)
        seeds = np.asarray(z["train_seeds"]).astype(np.int64)
        stored = np.asarray(z["train_E"])
        clauses = []
        energies = []
        for i, seed in enumerate(seeds.tolist()):
            _m, cl, E = gen_instance(n, int(seed), compute_E=True)
            if not np.array_equal(E.astype(np.int16), stored[i].astype(np.int16)):
                raise RuntimeError(f"n{n} row {i} seed {seed} did not regenerate train_E")
            clauses.append(cl)
            energies.append(E.astype(np.int16))
        m = int(round(9.93 * n))
        cls_v, _cls_s = pack_clauses(clauses, m, n)
        E = np.stack(energies, axis=0)
        pi = pack_batch(cls_v, n, True)
        Ep = permute_energy_instances(E, pi)
        if not np.array_equal((Ep == 0).sum(axis=1), (E == 0).sum(axis=1)):
            raise RuntimeError(f"n{n}: packing changed the number of solutions")
        out = dst / f"train_n{n}.npz"
        np.savez_compressed(
            out,
            n=n,
            m=m,
            train_seeds=seeds,
            train_E=Ep.astype(np.int16),
            train_E_original=E.astype(np.int16),
            pi_max=pi.astype(np.int16),
            source=f"cap_pi_max_of_{src}",
        )
        log(f"packed train n{n} N={len(seeds)} → {out}")


def _run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log("RUN " + " ".join(cmd))
    with log_path.open("a") as f:
        f.write(f"\n=== {' '.join(cmd)} @ {time.strftime('%Y-%m-%d %H:%M:%S %z')} ===\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise SystemExit(f"command failed ec={p.returncode}: {cmd[1]}")


def train_walsh(out: Path, packs: Path, threads: int) -> None:
    for seed in SEEDS:
        pt = out / "walsh_L2" / f"params_L2_seed{seed}_B1000.pt"
        if pt.exists():
            log(f"walsh seed {seed} exists, skip train")
            continue
        _run(
            [
                PY, str(_SCR / "train69_ucx_walsh_degree.py"),
                "--L", "2", "--seed", str(seed),
                "--threads", str(threads),
                "--max_train", "100",
                "--eval_B", "1000",
                "--skip_eval",
                "--packs_dir", str(packs),
                "--out_dir", str(out / "walsh_L2"),
                "--w6", "1", "--w9", "2",
                "--adam_steps", "400",
                "--lbfgs_rounds", "8",
                "--batch_per_n", "32",
            ],
            out / "logs" / f"walsh_L2_seed{seed}.log",
        )


def train_power(out: Path, packs: Path, threads: int) -> None:
    for seed in SEEDS:
        pt = out / "power_p" / f"params_L16_seed{seed}.pt"
        if pt.exists():
            log(f"power_p seed {seed} exists, skip train")
            continue
        _run(
            [
                PY, str(_SCR / "train_beat_train69.py"),
                "--idea", "power_p",
                "--L", "16", "--seed", str(seed),
                "--threads", str(threads),
                "--batch_per_n", "32",
                "--w6", "1", "--w9", "2",
                "--packs_dir", str(packs),
                "--out_root", str(out),
                "--qaoa_dir", str(POWER_QAOA),
                "--eval_ns", "6,9,12,15",
                "--beat_mode", "raw",
                "--skip_eval",
            ],
            out / "logs" / f"power_L16_seed{seed}.log",
        )


def _angles_from_state(which: str, state: dict, n: int, device):
    """Same angle map as the frozen champs, from a freshly trained state."""
    if which == "walsh_L2":
        gamma = state["gamma_w"].to(device)
        phi = phi_from_c(state["c1"].to(device), state["c2"].to(device), state["c3"].to(device))
        return 2, gamma, phi
    p = float(state["p"])
    a = (8.0 / float(n - 1)) ** p
    gamma = state["gamma"].to(device)
    phi = phi_from_c(state["c1"].to(device), state["c2"].to(device) * a, state["c3"].to(device) * a)
    return 16, gamma, phi


def eval_family(which: str, pts: list[Path], B: int, out_json: Path) -> dict:
    device = torch.device("cpu")
    # strategies once
    strat = {}
    energies = {}
    for n in EVAL_NS:
        E, E_t, cls = load_core(n, B)
        st, _scores, _w = strategies_for_n(cls, n)
        strat[n] = st
        energies[n] = (E, E_t)
    rows = {}
    if out_json.exists():
        rows = json.loads(out_json.read_text())
    for pt in pts:
        state = torch.load(pt, map_location="cpu", weights_only=False)
        seed = int(state.get("seed", -1))
        key = f"seed{seed}"
        rows.setdefault(key, {})
        for labeling in ("pi_max", "identity"):
            if labeling in rows[key] and all(str(n) in rows[key][labeling] for n in EVAL_NS):
                log(f"eval skip {which} {key} {labeling}")
                continue
            rows[key].setdefault(labeling, {})
            for n in EVAL_NS:
                if str(n) in rows[key][labeling]:
                    continue
                L, gamma, phi = _angles_from_state(which, state, n, device)
                E, E_t = energies[n]
                if labeling == "identity":
                    E_use = E_t.detach().cpu().numpy()
                else:
                    E_use = permute_energy_instances(E, strat[n]["pi_max"])
                log(f"eval {which} {key} {labeling} n{n}")
                sm = eval_snap(n, L, gamma, phi, E_use, every=0)
                q = PUBLISHED[which]["qaoa"][n]
                sm["QAOA_Med"] = q
                sm["R"] = sm["Med"] / q
                if which == "power_L16":
                    sm["FourierQAOA_Med"] = FOURIER_L16[n]
                    sm["R_fourier"] = sm["Med"] / FOURIER_L16[n]
                rows[key][labeling][str(n)] = sm
                out_json.write_text(json.dumps(rows, indent=2) + "\n")
                log(f"  Med={sm['Med']:.4f} R={sm['R']:.4f}")
    out_json.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def write_retrain_md(out: Path, walsh: dict, power: dict) -> None:
    lines = [
        "# CAP retrain — packed train{6,9}",
        "",
        "Clause-aware cavity compilation. Angles are retrained on f'(x) = f(π_max^{-1} x).",
        "Reported Med uses that same relabeling at eval, so success is the original formula.",
        "Identity columns apply the new angles with no relabeling.",
        "",
        "## L=2 Walsh vs seed1 knee",
        "",
        "Frozen Walsh seed1 (identity labeling, sealed QAOA) knee R = **0.9567** at n15.",
        "Protocol match: max_train=100, Adam 400, LBFGS 8 rounds, w6=1, w9=2, seeds 0,1,2.",
        "",
        "| seed | labeling | n6 R | n9 R | n12 R | n15 R | n15 Med | knee R |",
        "|------|----------|------|------|-------|-------|---------|--------|",
    ]
    # frozen packed reference from smoke is added by the caller via files; compute from walsh dict only
    for key in sorted(walsh):
        for labeling in ("pi_max", "identity"):
            block = walsh[key][labeling]
            rs = [block[str(n)]["R"] for n in EVAL_NS]
            lines.append(
                f"| {key} | {labeling} | "
                + " | ".join(f"{r:.4f}" for r in rs)
                + f" | {block['15']['Med']:.4f} | {max(rs):.4f} |"
            )
    lines += [
        "",
        "Beat seed1 means packed n15 R < 0.9567 (lower Med / same QAOA).",
        "",
        "## L=16 power_p vs fresh250 champ and Fourier QAOA",
        "",
        "Frozen power_p identity n15 Med = **2.5791**. Fourier QAOA Q=6 n15 Med = **2.3853**.",
        "Protocol match: full train 250/250, Adam 600, LBFGS 12, fresh250 QAOA angles, seeds 0,1,2.",
        "",
        "| seed | labeling | n15 Med | vs power_p 2.579 | vs Fourier 2.385 | n12 Med |",
        "|------|----------|---------|-------------------|------------------|---------|",
    ]
    for key in sorted(power):
        for labeling in ("pi_max", "identity"):
            b15 = power[key][labeling]["15"]
            b12 = power[key][labeling]["12"]
            lines.append(
                f"| {key} | {labeling} | {b15['Med']:.4f} | "
                f"{b15['Med'] - 2.579069137573242:+.4f} | "
                f"{b15['Med'] - FOURIER_L16[15]:+.4f} | {b12['Med']:.4f} |"
            )
    lines.append("")
    (out / "RETRAIN.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {out / 'RETRAIN.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(S.OUT / "beat_qaoa_12h" / "cap" / "retrain"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--phase", choices=["all", "build", "train", "eval"], default="all")
    ap.add_argument("--only", choices=["", "walsh", "power"], default="")
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    walsh_packs = out / "packs_walsh"
    power_packs = out / "packs_power"
    do_w = args.only in ("", "walsh")
    do_p = args.only in ("", "power")
    if args.phase in ("all", "build"):
        if do_w:
            build_packed(WALSH_SRC, walsh_packs)
        if do_p:
            build_packed(POWER_SRC, power_packs)
    if args.phase in ("all", "train"):
        if do_w:
            train_walsh(out, walsh_packs, args.threads)
        if do_p:
            train_power(out, power_packs, args.threads)
    walsh_rows, power_rows = {}, {}
    if args.phase in ("all", "eval"):
        if do_w:
            pts = [out / "walsh_L2" / f"params_L2_seed{s}_B1000.pt" for s in SEEDS]
            walsh_rows = eval_family("walsh_L2", pts, args.B, out / "walsh_eval.json")
        if do_p:
            pts = [out / "power_p" / f"params_L16_seed{s}.pt" for s in SEEDS]
            power_rows = eval_family("power_L16", pts, args.B, out / "power_eval.json")
        if do_w and do_p:
            write_retrain_md(out, walsh_rows, power_rows)
        elif do_w and (out / "power_eval.json").exists():
            write_retrain_md(out, walsh_rows, json.loads((out / "power_eval.json").read_text()))
        elif do_p and (out / "walsh_eval.json").exists():
            write_retrain_md(out, json.loads((out / "walsh_eval.json").read_text()), power_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

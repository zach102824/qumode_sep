#!/usr/bin/env python3
"""Train ideal qubit QAOA L=1 on sealed train{9,12}, eval scale1000 B=1000.

Angles: shared (β, γ) single-layer, trained mean−log p_succ on train packs (same
spirit as Arm C zero-shot), then frozen eval on data/scale1000.

Writes:
  explore_OCT29/baselines/scale1000/qaoa_L1_RESULTS.md
  explore_OCT29/baselines/scale1000/qaoa_L1_results.json
  explore_OCT29/runs/qaoa_L1_scale1000/  (angles, logs, per-n inv)
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

from explore_OCT29.lib.drivers import fit_b  # noqa: E402
from explore_OCT29.lib.qaoa import (  # noqa: E402
    clause_count,
    grow_angles,
    inv_p_from_psucc,
    linear_ramp_angles,
    paper_k4_p10_mapped,
    qaoa_psucc,
    summarize_inv,
)
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402

OUT = _PKG / "runs" / "qaoa_L1_scale1000"
BASE = _PKG / "baselines" / "scale1000"
DATA_SCALE = _PKG / "data" / "scale1000"
DATA_SEALED = _PKG / "data"

# Arm C L=1 reference (explore_OCT29/runs/armC_L1_scale1000/RESULTS.md)
ARM_C = {
    9: {"median_inv_p": 47.469, "B90_inv_p": 99.340, "B": 1000},
    12: {"median_inv_p": 194.177, "B90_inv_p": 425.207, "B": 1000},
    15: {"median_inv_p": 793.389, "B90_inv_p": 1870.020, "B": 1000},
    18: {"median_inv_p": 2993.655, "B90_inv_p": 8178.549, "B": 1000},
}

# Nov progressive shell L=1 (Nov/runs/angle_explore/L_LADDER_scale1000.md)
NOV_SHELL = {
    9: {"median_inv_p": 43.410, "B90_inv_p": 91.832, "B": 1000},
    12: {"median_inv_p": 172.318, "B90_inv_p": 382.556, "B": 1000},
    15: {"median_inv_p": 676.600, "B90_inv_p": 1643.524, "B": 1000},
    18: {"median_inv_p": 2541.414, "B90_inv_p": 6948.326, "B": 1000},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[qaoa_L1]", *a, flush=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        x = float(o)
        if math.isnan(x) or math.isinf(x):
            return x
        return x
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


def _S(med, b90):
    if b90 is None or not np.isfinite(b90) or not np.isfinite(med):
        return float("nan")
    return 0.5 * (math.log(max(med, 1e-12)) + math.log(max(b90, 1e-12)))


def load_E_split(n: int, data_dir: Path, split: str, max_count: int | None):
    pack = load_pack(n, data_dir=data_dir)
    if f"{split}_seeds" not in pack:
        raise RuntimeError(f"n={n}: missing split {split} in {data_dir}")
    seeds, _, _, _, E = _split_arrays(pack, split, max_count=max_count)
    if E is None or len(seeds) == 0:
        raise RuntimeError(f"n={n}: no dense E for split={split}")
    return seeds, np.asarray(E, dtype=np.float32)


def make_starts(L: int, nstart: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Each start is length 2L: [betas | gammas]."""
    starts = []
    b10, g10 = paper_k4_p10_mapped()
    b, g = grow_angles(b10, g10, L)
    starts.append(np.concatenate([b, g]))
    for s in (0.5, 1.0, 2.0, 4.0):
        bb, gg = linear_ramp_angles(L, scale=s)
        starts.append(np.concatenate([bb, gg]))
    # sign-flipped gammas (paper map uses negative γ)
    bb, gg = linear_ramp_angles(L, scale=1.0)
    starts.append(np.concatenate([bb, -gg]))
    while len(starts) < nstart:
        bb = rng.uniform(0.0, math.pi, size=L)
        gg = rng.uniform(-math.pi, math.pi, size=L)
        starts.append(np.concatenate([bb, gg]))
    return starts[:nstart]


def train_angles(
    L: int,
    train_ns: list[int],
    n_per: dict[int, int],
    data_dir: Path,
    split: str,
    device: torch.device,
    nstart: int,
    steps: int,
    lr: float,
    seed: int,
    chunk: int,
) -> dict:
    """Adam multi-start on mean −log p_succ across train_ns (equal weight per n)."""
    rng = np.random.default_rng(seed)
    bundles = {}
    for n in train_ns:
        seeds, E = load_E_split(n, data_dir, split, n_per.get(n))
        bundles[n] = torch.as_tensor(E, dtype=torch.float32, device=device)
        log(f"train load n={n} B={E.shape[0]} dim={E.shape[1]}")

    starts = make_starts(L, nstart, rng)
    best = None
    hist_all = []

    for si, x0 in enumerate(starts):
        betas = torch.nn.Parameter(torch.tensor(x0[:L], dtype=torch.float32, device=device))
        gammas = torch.nn.Parameter(torch.tensor(x0[L:], dtype=torch.float32, device=device))
        opt = torch.optim.Adam([betas, gammas], lr=lr)
        t0 = time.time()
        hist = []
        for step in range(steps):
            opt.zero_grad()
            losses = []
            for n, Efull in bundles.items():
                B = Efull.shape[0]
                acc = []
                for s in range(0, B, chunk):
                    Eb = Efull[s : s + chunk]
                    ps = qaoa_psucc(betas, gammas, Eb, n)
                    acc.append(-torch.log(ps.clamp_min(1e-12)).mean())
                losses.append(torch.stack(acc).mean())
            loss = torch.stack(losses).mean()
            loss.backward()
            opt.step()
            hist.append(float(loss.detach().cpu()))
            if step % 20 == 0 or step == steps - 1:
                log(
                    f"start {si+1}/{len(starts)} step {step:3d} "
                    f"loss={hist[-1]:.4f} wall={time.time()-t0:.0f}s"
                )
        final = hist[-1]
        payload = {
            "start_idx": si,
            "loss": final,
            "betas": betas.detach().cpu().numpy().tolist(),
            "gammas": gammas.detach().cpu().numpy().tolist(),
            "history": hist,
            "wall_s": time.time() - t0,
        }
        hist_all.append({"start_idx": si, "loss": final, "wall_s": payload["wall_s"]})
        if best is None or final < best["loss"]:
            best = payload
            log(f"new best start={si} loss={final:.4f}")

    return {
        "L": L,
        "train_ns": train_ns,
        "n_per": n_per,
        "split": split,
        "data_dir": str(data_dir),
        "nstart": nstart,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "best": best,
        "all_starts": hist_all,
        "angle_convention": "qaoa_ref: U_C=exp(-i γ Cn), U_B=exp(-i β X/2), Cn=E*n/m",
        "init_notes": "starts include paper k4 p=10 grown→L, linear ramps, random",
    }


@torch.no_grad()
def eval_n(
    n: int,
    betas: torch.Tensor,
    gammas: torch.Tensor,
    data_dir: Path,
    split: str,
    max_count: int,
    device: torch.device,
    chunk: int,
    progress_every: int,
) -> dict:
    seeds, E = load_E_split(n, data_dir, split, max_count)
    B = len(seeds)
    invs = []
    t0 = time.time()
    for s in range(0, B, chunk):
        Eb = torch.as_tensor(E[s : s + chunk], dtype=torch.float32, device=device)
        ps = qaoa_psucc(betas, gammas, Eb, n).cpu().numpy()
        invs.extend(inv_p_from_psucc(ps).tolist())
        done = min(s + chunk, B)
        if done % progress_every < chunk or done == B:
            med = float(np.median(invs))
            rate = done / max(time.time() - t0, 1e-9)
            eta = (B - done) / max(rate, 1e-9)
            log(f"eval n={n} {done}/{B} Med≈{med:.3f} rate={rate:.2f}/s ETA={eta/60:.1f}min")
    summ = summarize_inv(invs)
    summ.update(
        {
            "n": n,
            "split": split,
            "L": int(betas.numel()),
            "L_times_Med": summ["median_inv_p"] * int(betas.numel())
            if np.isfinite(summ["median_inv_p"])
            else float("inf"),
            "S": _S(summ["median_inv_p"], summ["B90_inv_p"]),
            "wall_s": time.time() - t0,
            "m": clause_count(n),
        }
    )
    return summ, np.asarray(invs, dtype=np.float64)


def fit_windows(ladder: dict) -> dict:
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
            if not m:
                continue
            if m.get("B90_inv_p") is None or not np.isfinite(m["B90_inv_p"]):
                continue
            ns_u.append(int(n))
            meds.append(m["median_inv_p"])
            b90s.append(m["B90_inv_p"])
        if len(ns_u) < 2:
            # still try Med-only for partial
            ns_u, meds = [], []
            for n in ns:
                m = ladder.get(n) or ladder.get(str(n))
                if m and np.isfinite(m.get("median_inv_p", np.nan)):
                    ns_u.append(int(n))
                    meds.append(m["median_inv_p"])
            if len(ns_u) >= 2:
                out[name] = {"ns": ns_u, "b_Med": fit_b(ns_u, meds), "b_B90": None}
            continue
        out[name] = {
            "ns": ns_u,
            "b_Med": fit_b(ns_u, meds),
            "b_B90": fit_b(ns_u, b90s),
        }
    return out


def write_results_md(path: Path, meta: dict, ladder: dict, fits: dict):
    def fmt(d, k, digits=3):
        v = d.get(k) if isinstance(d, dict) else None
        if v is None:
            return "—"
        try:
            x = float(v)
            if not np.isfinite(x):
                return "—"
            return f"{x:.{digits}f}"
        except Exception:
            return "—"

    def p_gs(med):
        try:
            x = float(med)
            if not np.isfinite(x) or x <= 0:
                return "—"
            return f"{1.0/x:.6f}"
        except Exception:
            return "—"

    L = int(meta.get("L", 1))
    lines = [
        f"# QAOA L={L} — scale1000 B=1000",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        f"**Ansatz:** ideal qubit QAOA `|+>^n → [U_C(γ) U_B(β)]^L`, L={L}",
        "**Cost:** `Cn = E·(n/m)`, `U_C = exp(-i γ Cn)` (unsat count / SAT Hamiltonian)",
        "**Mixer:** `U_B = ⊗ exp(-i β X/2)` (var0 = MSB of E index)",
        f"**Angles:** trained shared (β,γ) on sealed `{meta['train_split']}` "
        f"n={meta['train_ns']} (n_per={meta['n_per']}), zero-shot eval",
        f"**Pack:** `{meta['eval_data_dir']}` split=`{meta['eval_split']}`",
        f"**Angles file:** `{meta.get('angles_path', '')}`",
        "",
        "## Ladder (1/p)",
        "",
        "| n | Med | B90 | p(gs)≈1/Med | L×Med | frac_inv≤2 | B | wall_min |",
        "|---|-----|-----|-------------|-------|------------|---|----------|",
    ]
    for n in sorted(int(k) for k in ladder.keys()):
        m = ladder[n] if n in ladder else ladder[str(n)]
        b90 = m["B90_inv_p"]
        b90_s = f"{b90:.3f}" if b90 is not None and np.isfinite(b90) else "—"
        lines.append(
            f"| {n} | {m['median_inv_p']:.3f} | {b90_s} | "
            f"{p_gs(m['median_inv_p'])} | "
            f"{m['L_times_Med']:.1f} | {m['frac_inv_le_2']:.3f} | "
            f"{m['B']} | {m['wall_s']/60:.1f} |"
        )
    lines += [
        "",
        "## vs Arm C L=1 + Nov progressive shell L=1 (same packs)",
        "",
        "| n | QAOA Med | QAOA B90 | QAOA p(gs) | ArmC Med | ArmC B90 | ArmC p(gs) | NovShell Med | NovShell B90 | NovShell p(gs) |",
        "|---|----------|----------|------------|----------|----------|------------|--------------|--------------|----------------|",
    ]
    for n in sorted(set(list(ladder.keys()) + list(ARM_C.keys()) + list(NOV_SHELL.keys()))):
        n = int(n)
        q = ladder.get(n) or ladder.get(str(n)) or {}
        a = ARM_C.get(n, {})
        s = NOV_SHELL.get(n, {})
        lines.append(
            f"| {n} | {fmt(q,'median_inv_p')} | {fmt(q,'B90_inv_p')} | {p_gs(q.get('median_inv_p'))} | "
            f"{fmt(a,'median_inv_p')} | {fmt(a,'B90_inv_p')} | {p_gs(a.get('median_inv_p'))} | "
            f"{fmt(s,'median_inv_p')} | {fmt(s,'B90_inv_p')} | {p_gs(s.get('median_inv_p'))} |"
        )
    lines += [
        "",
        "## Side-by-side Med (1/p) only",
        "",
        "| n | QAOA L=1 | Arm C L=1 | Nov shell L=1 | QAOA / ArmC | QAOA / Nov |",
        "|---|----------|-----------|---------------|-------------|------------|",
    ]
    for n in (9, 12, 15, 18):
        q = ladder.get(n) or ladder.get(str(n)) or {}
        a = ARM_C.get(n, {})
        s = NOV_SHELL.get(n, {})
        qm = q.get("median_inv_p")
        am = a.get("median_inv_p")
        sm = s.get("median_inv_p")
        def ratio(num, den):
            try:
                return f"{float(num)/float(den):.3f}×"
            except Exception:
                return "—"
        lines.append(
            f"| {n} | {fmt(q,'median_inv_p')} | {fmt(a,'median_inv_p')} | {fmt(s,'median_inv_p')} | "
            f"{ratio(qm, am)} | {ratio(qm, sm)} |"
        )
    lines += ["", "## b fits", ""]
    for name, f in fits.items():
        bm = f["b_Med"]["b"] if f.get("b_Med") else float("nan")
        bb = f["b_B90"]["b"] if f.get("b_B90") else float("nan")
        lines.append(f"- **{name}** (ns={f['ns']}): b_Med≈**{bm:.4f}**, b_B90≈**{bb:.4f}**")
    lines += [
        "",
        "## Notes",
        "",
        "- Comparators: Arm C L=1 (`runs/armC_L1_scale1000`) and Nov progressive shell L=1 "
        "(`Nov/runs/angle_explore/L_LADDER_scale1000.md`).",
        "- QAOA angles trained from multi-start (paper p=10 grown→L, linear ramps, random).",
        "- p(gs)≈1/Med is the median ground-state success probability.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--L", type=int, default=1)
    ap.add_argument("--train_ns", type=str, default="9,12")
    ap.add_argument("--n_train_9", type=int, default=200)
    ap.add_argument("--n_train_12", type=int, default=128)
    ap.add_argument("--train_data", type=str, default=str(DATA_SEALED))
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--eval_data", type=str, default=str(DATA_SCALE))
    ap.add_argument("--eval_split", type=str, default="scale1000")
    ap.add_argument("--ns", type=str, default="9,12,15,18")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--B18", type=int, default=1000, help="cap n=18 instances")
    ap.add_argument("--nstart", type=int, default=12)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--train_chunk", type=int, default=64)
    ap.add_argument("--eval_chunk", type=int, default=0, help="0 → auto by n")
    ap.add_argument("--progress_every", type=int, default=25)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--angles", type=str, default="", help="load saved angles json/pt")
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)

    OUT.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    L = int(args.L)
    train_ns = [int(x) for x in args.train_ns.split(",") if x.strip()]
    n_per = {9: args.n_train_9, 12: args.n_train_12}
    eval_ns = [int(x) for x in args.ns.split(",") if x.strip()]

    angles_path = OUT / "angles.json"
    train_blob = None

    if args.angles:
        apath = Path(args.angles)
        train_blob = json.loads(apath.read_text())
        log(f"loaded angles from {apath}")
    elif args.skip_train or args.eval_only:
        if not angles_path.exists():
            raise SystemExit(f"no angles at {angles_path}; run train first")
        train_blob = json.loads(angles_path.read_text())
        log(f"loaded angles from {angles_path}")
    else:
        log("=== train angles ===")
        train_blob = train_angles(
            L,
            train_ns,
            n_per,
            Path(args.train_data),
            args.train_split,
            device,
            args.nstart,
            args.steps,
            args.lr,
            args.seed,
            args.train_chunk,
        )
        angles_path.write_text(json.dumps(_jsonable(train_blob), indent=2))
        log(f"wrote {angles_path} best_loss={train_blob['best']['loss']:.4f}")

    betas = torch.tensor(train_blob["best"]["betas"], dtype=torch.float32, device=device)
    gammas = torch.tensor(train_blob["best"]["gammas"], dtype=torch.float32, device=device)
    if betas.numel() != L:
        raise SystemExit(f"L mismatch angles {betas.numel()} vs {L}")

    meta = {
        "L": L,
        "train_ns": train_ns,
        "n_per": n_per,
        "train_split": args.train_split,
        "train_data_dir": args.train_data,
        "eval_split": args.eval_split,
        "eval_data_dir": args.eval_data,
        "B": args.B,
        "B18": args.B18,
        "angles_path": str(angles_path if not args.angles else args.angles),
        "train_loss": train_blob["best"]["loss"],
        "device": str(device),
        "threads": args.threads,
        "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "angle_convention": train_blob.get("angle_convention"),
    }

    results_path = OUT / "results.json"
    ladder = {}
    if results_path.exists():
        try:
            prev = json.loads(results_path.read_text())
            for k, v in (prev.get("ladder") or {}).items():
                ladder[int(k)] = v
            log(f"resumed ladder ns={sorted(ladder)}")
        except Exception as e:
            log("resume skip", e)

    for n in eval_ns:
        Bcap = args.B18 if n >= 18 else args.B
        if n in ladder and ladder[n].get("B", 0) >= Bcap:
            log(f"skip n={n} already B={ladder[n]['B']}")
            continue
        chunk = args.eval_chunk
        if chunk <= 0:
            chunk = 64 if n <= 9 else (16 if n <= 12 else (4 if n <= 15 else 1))
        log(f"=== eval n={n} B<={Bcap} chunk={chunk} ===")
        summ, invs = eval_n(
            n,
            betas,
            gammas,
            Path(args.eval_data),
            args.eval_split,
            Bcap,
            device,
            chunk,
            args.progress_every,
        )
        ladder[n] = summ
        np.save(OUT / f"inv_p_n{n}.npy", invs)
        log(
            f"DONE n={n}: Med={summ['median_inv_p']:.3f} B90={summ['B90_inv_p']:.3f} "
            f"B={summ['B']} wall={summ['wall_s']/60:.1f}min"
        )
        fits = fit_windows(ladder)
        blob = {
            "meta": meta,
            "ladder": {str(k): v for k, v in sorted(ladder.items())},
            "fits": _jsonable(fits),
            "arm_C_ref": ARM_C,
            "nov_shell_L1_ref": NOV_SHELL,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        }
        results_path.write_text(json.dumps(_jsonable(blob), indent=2))
        write_results_md(OUT / "RESULTS.md", meta, ladder, fits)
        write_results_md(BASE / "qaoa_L1_RESULTS.md", meta, ladder, fits)
        (BASE / "qaoa_L1_results.json").write_text(json.dumps(_jsonable(blob), indent=2))

    fits = fit_windows(ladder)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    blob = {
        "meta": meta,
        "ladder": {str(k): v for k, v in sorted(ladder.items())},
        "fits": _jsonable(fits),
        "arm_C_ref": ARM_C,
            "nov_shell_L1_ref": NOV_SHELL,
        "updated": meta["finished"],
    }
    results_path.write_text(json.dumps(_jsonable(blob), indent=2))
    write_results_md(OUT / "RESULTS.md", meta, ladder, fits)
    write_results_md(BASE / "qaoa_L1_RESULTS.md", meta, ladder, fits)
    (BASE / "qaoa_L1_results.json").write_text(json.dumps(_jsonable(blob), indent=2))
    log("ALL DONE")
    for name, f in fits.items():
        bm = f["b_Med"]["b"] if f.get("b_Med") else float("nan")
        bb = f["b_B90"]["b"] if f.get("b_B90") else float("nan")
        log(f"fit {name}: b_Med={bm:.4f} b_B90={bb:.4f}")


if __name__ == "__main__":
    main()

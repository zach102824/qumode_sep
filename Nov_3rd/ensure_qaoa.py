"""Ensure paper-init QAOA angles + scale1000 refs exist for a given L.

Fair same-L baseline for Nov_3rd: train qubit QAOA on n=6×1000 from
Boulebnane–Montanaro paper init (β=0.01, γ=−0.01, nstart=1).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from Nov_3rd import settings as S
from Nov_3rd.data_io import log
from Nov_3rd.qaoa_init import (
    ANGLE_CONVENTION,
    INIT_NOTES,
    QAOA_NSTART,
    make_starts,
)
from explore_OCT29.lib.qaoa import (
    clause_count,
    inv_p_from_psucc,
    qaoa_psucc,
    summarize_inv,
)
from explore_OCT29.lib.train_loop import _split_arrays, load_pack


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


def _is_paper_n6(angles_path: Path) -> bool:
    if not angles_path.exists():
        return False
    try:
        data = json.loads(angles_path.read_text())
    except Exception:
        return False
    train_ns = data.get("train_ns") or []
    if isinstance(train_ns, dict):
        train_ns = list(train_ns.keys())
    train_ns = [int(x) for x in train_ns]
    nstart = int(data.get("nstart") or 0)
    notes = str(data.get("init_notes") or "") + str(data.get("purpose") or "")
    paperish = ("2208.06909" in notes) or ("Boulebnane" in notes) or ("paper" in notes.lower())
    return train_ns == [6] and nstart == QAOA_NSTART and paperish and "best" in data


def _load_E(n: int, data_dir: Path, split: str, max_count: int | None):
    pack = load_pack(n, data_dir=data_dir)
    seeds, _, _, _, E = _split_arrays(pack, split, max_count=max_count)
    if E is None or len(seeds) == 0:
        raise RuntimeError(f"n={n}: no dense E for split={split} in {data_dir}")
    return seeds, np.asarray(E, dtype=np.float32)


def train_paper_n6(
    L: int,
    *,
    device: torch.device,
    steps: int | None = None,
    lr: float | None = None,
    seed: int = 0,
    chunk: int = 64,
) -> dict:
    """Adam train QAOA from paper init on sealed n6×1000."""
    L = int(L)
    # scale budget mildly with L (paper nstart=1 only)
    if steps is None:
        steps = 200 if L <= 4 else (300 if L <= 16 else 400)
    if lr is None:
        lr = 0.08 if L <= 4 else 0.05
    seeds, E = _load_E(6, S.SEALED_DATA, "train", S.MAX_TRAIN)
    E_t = torch.as_tensor(E, dtype=torch.float32, device=device)
    log(f"QAOA L{L} paper-init train n6 B={E_t.shape[0]} steps={steps} lr={lr}")
    x0 = make_starts(L, QAOA_NSTART)[0]
    betas = torch.nn.Parameter(torch.tensor(x0[:L], dtype=torch.float32, device=device))
    gammas = torch.nn.Parameter(torch.tensor(x0[L:], dtype=torch.float32, device=device))
    opt = torch.optim.Adam([betas, gammas], lr=lr)
    t0 = time.time()
    hist = []
    for step in range(steps):
        opt.zero_grad()
        acc = []
        B = E_t.shape[0]
        for s0 in range(0, B, chunk):
            Eb = E_t[s0 : s0 + chunk]
            ps = qaoa_psucc(betas, gammas, Eb, 6)
            acc.append(-torch.log(ps.clamp_min(1e-12)).mean())
        loss = torch.stack(acc).mean()
        loss.backward()
        opt.step()
        hist.append(float(loss.detach().cpu()))
        if step % 25 == 0 or step == steps - 1:
            log(
                f"QAOA L{L} step {step:3d}/{steps} loss={hist[-1]:.4f} "
                f"wall={time.time()-t0:.0f}s"
            )
    best = {
        "start_idx": 0,
        "loss": hist[-1],
        "betas": betas.detach().cpu().tolist(),
        "gammas": gammas.detach().cpu().tolist(),
        "history": hist,
        "wall_s": time.time() - t0,
    }
    return {
        "L": L,
        "train_ns": [6],
        "n_per": {"6": int(E_t.shape[0])},
        "split": "train",
        "data_dir": str(S.SEALED_DATA),
        "nstart": QAOA_NSTART,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "best": best,
        "all_starts": [{"start_idx": 0, "loss": best["loss"], "wall_s": best["wall_s"]}],
        "angle_convention": ANGLE_CONVENTION,
        "init_notes": INIT_NOTES,
        "purpose": "fair baseline: same train budget as Nov_3rd (n=6×1000)",
    }


@torch.no_grad()
def eval_scale_ladder(
    L: int,
    betas: torch.Tensor,
    gammas: torch.Tensor,
    *,
    device: torch.device,
    ns=None,
    max_count: int = S.SCALE_B,
) -> dict:
    if ns is None:
        ns = list(S.EVAL_NS_SIM)
    ladder = {}
    for n in ns:
        pack = S.SCALE1000_DIR / f"data_n{n}.npz"
        if not pack.exists():
            log(f"QAOA L{L} skip n{n}: missing pack")
            continue
        seeds, E = _load_E(n, S.SCALE1000_DIR, "scale1000", max_count)
        B = len(seeds)
        chunk = 64 if n <= 9 else (16 if n <= 12 else (4 if n <= 15 else 1))
        invs = []
        t0 = time.time()
        for s0 in range(0, B, chunk):
            Eb = torch.as_tensor(E[s0 : s0 + chunk], dtype=torch.float32, device=device)
            ps = qaoa_psucc(betas, gammas, Eb, n).cpu().numpy()
            invs.extend(inv_p_from_psucc(ps).tolist())
            done = min(s0 + chunk, B)
            if done % max(chunk, 50) < chunk or done == B:
                rate = done / max(time.time() - t0, 1e-9)
                eta = (B - done) / max(rate, 1e-9)
                log(
                    f"QAOA L{L} eval n{n} {done}/{B} Med≈{float(np.median(invs)):.3f} "
                    f"ETA={eta/60:.1f}min"
                )
        summ = summarize_inv(invs)
        med, b90 = summ["median_inv_p"], summ["B90_inv_p"]
        import math

        Slog = (
            0.5 * (math.log(max(med, 1e-12)) + math.log(max(b90, 1e-12)))
            if np.isfinite(med) and np.isfinite(b90)
            else float("nan")
        )
        summ.update(
            {
                "n": n,
                "split": "scale1000",
                "L": L,
                "S": Slog,
                "m": clause_count(n),
                "wall_s": time.time() - t0,
                "source": "fresh",
            }
        )
        ladder[n] = summ
        log(f"QAOA L{L} n{n}: Med={med:.3f} B90={b90:.3f} B={summ['B']}")
    return ladder


def ensure_qaoa_paper_n6(L: int, *, device: torch.device | None = None, force: bool = False) -> Path:
    """Train+eval paper-init QAOA into qaoa_angles/L{L}/ if missing or not paper-n6."""
    L = int(L)
    out = S.QAOA_DIR / f"L{L}"
    out.mkdir(parents=True, exist_ok=True)
    angles_path = out / "angles.json"
    results_path = out / "results.json"
    device = device or torch.device("cpu")

    need_train = force or not _is_paper_n6(angles_path)
    if need_train:
        # archive non-paper angles once
        if angles_path.exists() and not _is_paper_n6(angles_path):
            legacy = S.QAOA_DIR / f"L{L}_legacy_nonpaper"
            if not legacy.exists():
                legacy.mkdir(parents=True, exist_ok=True)
                for name in ("angles.json", "results.json"):
                    src = out / name
                    if src.exists():
                        (legacy / name).write_bytes(src.read_bytes())
                log(f"QAOA L{L}: archived non-paper → {legacy}")
        blob = train_paper_n6(L, device=device)
        angles_path.write_text(json.dumps(_jsonable(blob), indent=2) + "\n")
        log(f"QAOA L{L}: wrote {angles_path} loss={blob['best']['loss']:.4f}")
    else:
        blob = json.loads(angles_path.read_text())
        log(f"QAOA L{L}: reuse paper-n6 angles @ {angles_path}")

    # ensure scale1000 results cover EVAL_NS_SIM
    need_eval = force or not results_path.exists()
    existing = {}
    if results_path.exists() and not force:
        try:
            existing = json.loads(results_path.read_text())
            ladder = {int(k): v for k, v in (existing.get("ladder") or {}).items()}
            if all(n in ladder for n in S.EVAL_NS_SIM):
                need_eval = False
            else:
                need_eval = True
        except Exception:
            need_eval = True

    if need_eval:
        betas = torch.tensor(blob["best"]["betas"], dtype=torch.float32, device=device)
        gammas = torch.tensor(blob["best"]["gammas"], dtype=torch.float32, device=device)
        # fill only missing ns when partial results exist
        have = set()
        if results_path.exists() and not force:
            try:
                have = {int(k) for k in (json.loads(results_path.read_text()).get("ladder") or {})}
            except Exception:
                have = set()
        missing = [n for n in S.EVAL_NS_SIM if n not in have]
        if force:
            missing = list(S.EVAL_NS_SIM)
        log(f"QAOA L{L}: eval scale1000 ns={missing}")
        fresh = eval_scale_ladder(L, betas, gammas, device=device, ns=missing)
        if results_path.exists() and not force:
            old = json.loads(results_path.read_text())
            ladder = {int(k): v for k, v in (old.get("ladder") or {}).items()}
            ladder.update(fresh)
            meta = old.get("meta") or {}
        else:
            ladder = dict(fresh)
            meta = {}
        meta.update(
            {
                "L": L,
                "train": f"n6×{S.MAX_TRAIN} sealed, paper init β=0.01 γ=-0.01 nstart=1",
                "betas": blob["best"]["betas"],
                "gammas": blob["best"]["gammas"],
                "train_loss": blob["best"]["loss"],
                "nstart": QAOA_NSTART,
                "init_notes": INIT_NOTES,
                "angle_convention": ANGLE_CONVENTION,
                "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            }
        )
        payload = {
            "meta": meta,
            "ladder": {str(k): v for k, v in sorted(ladder.items())},
            "updated": meta["finished"],
        }
        results_path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
        log(f"QAOA L{L}: wrote {results_path}")
    return angles_path

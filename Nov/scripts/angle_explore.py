#!/usr/bin/env python3
"""Nonstop angle-finding + SNAP-head explore (Nov1st-final).

Fixed: max_train=200, n∈{6,9}, max_val=40, L∈{1,4,...,32}, scale1000 B=1000.

SNAP heads (Zach add): shell, free, bits, fourier, proj, x_pop × shared|perL.

Stop: touch Nov/runs/STOP_ANGLE_EXPLORE
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_NOV = Path(__file__).resolve().parents[1]
_ROOT = _NOV.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.drivers import fit_b  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.binH_vs_grayH import build_W_BinH  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_explore import _jsonable  # noqa: E402
from explore_OCT29.scripts.probe6_spectral_mixer import _S  # noqa: E402
from Nov.scripts.nov1st_final_train import (  # noqa: E402
    SEALED_REF,
    eval_split,
    make_sim,
    scale1000_eval,
    smoke_overlap,
)
from Nov.scripts.snap_heads import (  # noqa: E402
    SNAP_MODES,
    describe_heads,
    init_snap_params,
    pad_snap,
    snap_dim,
    snap_tag,
)

OUT = _NOV / "runs" / "angle_explore"
STOP = _NOV / "runs" / "STOP_ANGLE_EXPLORE"
L_LADDER = (1, 4, 8, 12, 16, 20, 24, 28, 32)


def log(*a):
    print(time.strftime("%H:%M:%S"), "[angle-explore]", *a, flush=True)


def stopped() -> bool:
    return STOP.exists()


def loss_fn(outs, kind: str, leakpen: float, cvar_q: float = 0.8):
    invs = torch.stack([o["inv_p"] for o in outs])
    leaks = torch.stack([o["p_leak"] for o in outs])
    if kind == "log_inv":
        base = torch.log(invs.clamp(min=1.0))
    elif kind == "raw_inv":
        base = invs
    elif kind == "softplus":
        base = torch.nn.functional.softplus(torch.log(invs.clamp(min=1e-6)))
    elif kind == "cvar":
        vals = torch.log(invs.clamp(min=1.0))
        if vals.numel() == 1:
            return vals.mean() + leakpen * leaks.mean()
        k = max(1, int(math.ceil((1.0 - cvar_q) * vals.numel())))
        top = torch.topk(vals, k=k, largest=True).values
        return top.mean() + leakpen * leaks.mean()
    else:
        raise ValueError(kind)
    return base.mean() + leakpen * leaks.mean()


def pad_gamma(gamma: torch.Tensor | None, L_new: int) -> torch.Tensor:
    if gamma is None:
        return torch.linspace(0.3, 1.0, L_new)
    g = gamma.detach().clone().float().reshape(-1)
    L_old = g.numel()
    if L_new == L_old:
        return g
    if L_new < L_old:
        return g[:L_new]
    out = torch.zeros(L_new)
    out[:L_old] = g
    span = max(0.2, abs(float(g[-1] - g[0])) / max(L_old - 1, 1)) if L_old else 0.2
    for i, ell in enumerate(range(L_old, L_new)):
        out[ell] = float(g[-1]) + (i + 1) * 0.5 * span
    return out


def load_train_packs(train_ns, max_train, device):
    packs = {}
    for n in train_ns:
        pack = load_pack(n)
        _, _, _, _, E = _split_arrays(pack, "train", max_count=max_train)
        if E is None or len(E) == 0:
            _, _, _, _, E = _split_arrays(pack, "val", max_count=max_train)
        packs[n] = {"E": torch.as_tensor(E, dtype=torch.float32, device=device), "N": len(E)}
        log(f"pack n{n} train N={len(E)} (cap={max_train})")
    return packs


@torch.no_grad()
def eval_snap(
    n, W, gamma, snap, L, nf, device, *, snap_mode, snap_shared, fourier_M=2,
    split="val", max_count=40, data_dir=None, use_sealed=False, progress_every=0,
):
    pack = load_pack(n, data_dir=data_dir)
    key = split
    if f"{split}_seeds" not in pack:
        if "scale1000_seeds" in pack:
            key = "scale1000"
        elif "test_seeds" in pack:
            key = "test"
        else:
            return None
    _, _, _, _, E = _split_arrays(pack, key, max_count=max_count)
    if E is None or len(E) == 0:
        return None
    sim = make_sim(n, W, L, nf, device)
    invs, leaks = [], []
    t0 = time.time()
    B = len(E)
    for b in range(B):
        out = sim.forward_dispatch(
            torch.as_tensor(E[b], dtype=torch.float32, device=device),
            gamma, snap, use_sealed=use_sealed,
            snap_mode=snap_mode, snap_shared=snap_shared, fourier_M=fourier_M,
        )
        invs.append(float(out["inv_p"].detach().cpu()))
        leaks.append(float(out["p_leak"].detach().cpu()))
        if progress_every and (b + 1) % progress_every == 0:
            log(f"eval n{n} {key} {b+1}/{B} Med≈{np.median(invs):.3f}")
    invs = np.asarray(invs, dtype=np.float64)
    med = float(np.median(invs))
    b90 = float(np.quantile(invs, 0.90))
    return {
        "n": n, "split": key, "B": B,
        "median_inv_p": med, "B90_inv_p": b90,
        "mean_leak": float(np.mean(leaks)), "S": _S(med, b90),
        "L": L, "snap_mode": snap_mode, "snap_shared": snap_shared,
        "wall_s": time.time() - t0,
    }


def scale1000_snap(W, gamma, snap, L, nf, device, ns, *, snap_mode, snap_shared,
                   fourier_M=2, max_count=1000, use_sealed=False):
    from Nov.scripts.nov1st_final_train import SCALE_DIR
    out = {}
    if not SCALE_DIR.exists():
        log("scale1000 dir missing"); return out
    for n in ns:
        m = eval_snap(
            n, W, gamma, snap, L, nf, device,
            snap_mode=snap_mode, snap_shared=snap_shared, fourier_M=fourier_M,
            split="scale1000", max_count=max_count, data_dir=SCALE_DIR,
            use_sealed=use_sealed, progress_every=100 if n >= 15 else 200,
        )
        if m:
            out[n] = m
            ref = SEALED_REF.get(n)
            bit = ""
            if ref:
                bit = (f"  vs sealed Med {m['median_inv_p']/ref['Med']:.3f}× "
                       f"B90 {m['B90_inv_p']/ref['B90']:.3f}×")
            log(f"scale1000 n{n} L{L} {snap_mode}: Med={m['median_inv_p']:.3f} "
                f"B90={m['B90_inv_p']:.3f} B={m['B']}{bit}")
    return out


def train_one(
    W, L, nf, device, *,
    train_ns=(6, 9), max_train=200, max_val=40, seed=0, lr=2e-3, steps=400,
    use_sealed=False, gamma0=None, snap0=None,
    snap_mode="shell", snap_shared=False, fourier_M=2,
    loss_kind="log_inv", leakpen=0.1, cvar_q=0.8,
    lbfgs_mid=True, lbfgs_final=True, lbfgs_rounds=2, lbfgs_iter=6,
    val_every=20, minibatch=1, arm_tag="adam", out_dir: Path | None = None,
):
    train_ns = tuple(int(x) for x in train_ns)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    gamma = torch.nn.Parameter(pad_gamma(gamma0, L))
    if snap0 is None:
        snap_t = init_snap_params(snap_mode, L, shared=snap_shared, fourier_M=fourier_M)
    else:
        snap_t = pad_snap(snap0, L, snap_mode, snap_shared, fourier_M=fourier_M)
    snap = torch.nn.Parameter(snap_t.float())
    packs = load_train_packs(train_ns, max_train, device)
    sims = {n: make_sim(n, W, L, nf, device) for n in train_ns}
    opt = torch.optim.Adam([gamma, snap], lr=lr)
    best = {"score": float("inf"), "state": None, "val": None, "step": -1, "history": []}
    n_lo, n_hi = min(train_ns), max(train_ns)
    t0 = time.time()

    def fwd(n, Erow):
        return sims[n].forward_dispatch(
            Erow, gamma, snap, use_sealed=use_sealed,
            snap_mode=snap_mode, snap_shared=snap_shared, fourier_M=fourier_M,
        )

    def run_lbfgs(tag):
        opt_l = torch.optim.LBFGS(
            [gamma, snap], lr=0.15, max_iter=lbfgs_iter, history_size=12,
            line_search_fn="strong_wolfe",
        )
        def closure():
            opt_l.zero_grad()
            outs = []
            for n, P in packs.items():
                for _ in range(min(8, P["N"])):
                    idx = int(rng.integers(0, P["N"]))
                    outs.append(fwd(n, P["E"][idx]))
            lt = loss_fn(outs, loss_kind, leakpen, cvar_q)
            lt.backward()
            return lt
        try:
            opt_l.step(closure)
            log(f"  LBFGS {tag} ok")
        except Exception as e:
            log(f"  LBFGS {tag} skip: {e}")

    for step in range(steps):
        if stopped():
            log("STOP"); break
        opt.zero_grad()
        outs = []
        for n, P in packs.items():
            for _ in range(minibatch):
                idx = int(rng.integers(0, P["N"]))
                outs.append(fwd(n, P["E"][idx]))
        loss = loss_fn(outs, loss_kind, leakpen, cvar_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([gamma, snap], 5.0)
        opt.step()

        do_val = (step % val_every == 0) or (step + 1 == steps)
        if lbfgs_mid and step == steps // 2:
            run_lbfgs("mid"); do_val = True
        if lbfgs_final and step + 1 == steps:
            for r in range(lbfgs_rounds):
                run_lbfgs(f"final{r}")
            do_val = True

        if do_val:
            vals = {}
            for n in train_ns:
                vals[n] = eval_snap(
                    n, W, gamma.detach(), snap.detach(), L, nf, device,
                    snap_mode=snap_mode, snap_shared=snap_shared, fourier_M=fourier_M,
                    split="val", max_count=max_val, use_sealed=use_sealed,
                )
            sc = 0.4 * vals[n_lo]["S"] + 0.6 * vals[n_hi]["S"]
            best["history"].append({"step": step, "Scomb": sc})
            if sc < best["score"]:
                best = {
                    "score": sc, "step": step, "val": vals, "history": best["history"],
                    "state": {
                        "gamma": gamma.detach().cpu().clone(),
                        "snap": snap.detach().cpu().clone(),
                        # alias for shell loaders
                        "phi": snap.detach().cpu().clone() if snap_mode == "shell" and not snap_shared else None,
                        "L": L, "motif": "nov1st_final", "W": "BinH",
                        "train_ns": list(train_ns), "arm": arm_tag,
                        "snap_mode": snap_mode, "snap_shared": snap_shared,
                        "fourier_M": fourier_M, "loss_kind": loss_kind,
                        "leakpen": leakpen, "seed": seed, "lr": lr, "steps": steps,
                    },
                }
            bits = " ".join(
                f"n{n}:{vals[n]['median_inv_p']:.2f}/{vals[n]['B90_inv_p']:.2f}"
                for n in train_ns if vals.get(n)
            )
            log(f"{arm_tag} L={L} seed={seed} step {step}/{steps}: "
                f"S={sc:.3f} best={best['score']:.3f} {bits}")

    best["wall_s"] = time.time() - t0
    if out_dir is not None and best.get("state") is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{arm_tag}_L{L}_seed{seed}"
        pt = out_dir / f"params_{tag}.pt"
        torch.save(best["state"], pt)
        meta = {
            "score": best["score"], "step": best["step"], "wall_s": best["wall_s"],
            "val": {
                str(n): {"Med": best["val"][n]["median_inv_p"],
                         "B90": best["val"][n]["B90_inv_p"], "S": best["val"][n]["S"]}
                for n in train_ns if best.get("val") and best["val"].get(n)
            },
            "path": str(pt), "arm": arm_tag, "L": L, "seed": seed,
            "snap_mode": snap_mode, "snap_shared": snap_shared,
            "loss_kind": loss_kind, "leakpen": leakpen, "lr": lr, "steps": steps,
        }
        (out_dir / f"meta_{tag}.json").write_text(json.dumps(_jsonable(meta), indent=2))
        best["meta"] = meta
        best["path"] = str(pt)
    return best


def promote_kill(n9_med: float) -> str:
    if n9_med > 10.0:
        return "KILL"
    if n9_med <= 2.5:
        return "PROMOTE_CAND"
    return "KEEP"


def append_leaderboard(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fields = list(row.keys())
    # if file exists with different header, still append with union — keep simple
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def lb_row(arm, L, seed, lr, steps, loss, leakpen, best, snap_mode, snap_shared, path=""):
    n9 = best["val"][9]["median_inv_p"]
    return {
        "arm": arm, "L": L, "seed": seed, "lr": lr, "steps": steps,
        "loss": loss, "leakpen": leakpen, "snap_mode": snap_mode,
        "snap_shared": int(snap_shared), "Scomb": best["score"],
        "n6_Med": best["val"][6]["median_inv_p"], "n6_B90": best["val"][6]["B90_inv_p"],
        "n9_Med": n9, "n9_B90": best["val"][9]["B90_inv_p"],
        "verdict": promote_kill(n9), "path": path or best.get("path", ""),
        "wall_s": best.get("wall_s", 0),
    }


def run_progressive_shell(W, nf, device, *, seeds, lr, steps, max_train, max_val,
                          out_dir, start_L=1, warm_from=None):
    """Continue/complete progressive shell perL (baseline SNAP head)."""
    champs = {}
    prev = {s: (None, None) for s in seeds}  # gamma, snap
    if warm_from:
        for seed, (g, sn, L0) in warm_from.items():
            prev[seed] = (g, sn)
            log(f"warm seed{seed} from L{L0}")
    for L in L_LADDER:
        if L < start_L:
            continue
        if stopped():
            break
        log(f"=== progressive shell perL L={L} ===")
        best_L = None
        for seed in seeds:
            if stopped():
                break
            g0, s0 = prev[seed]
            arm = snap_tag("shell", False)
            best = train_one(
                W, L, nf, device, max_train=max_train, max_val=max_val, seed=seed,
                lr=lr, steps=steps, gamma0=g0, snap0=s0,
                snap_mode="shell", snap_shared=False,
                arm_tag=f"prog_{arm}", out_dir=out_dir / "progressive",
            )
            st = best["state"]
            prev[seed] = (st["gamma"], st["snap"])
            append_leaderboard(out_dir / "leaderboard.csv",
                               lb_row(f"prog_{arm}", L, seed, lr, steps, "log_inv", 0.1,
                                      best, "shell", False))
            if best_L is None or best["score"] < best_L["score"]:
                best_L = best
        champs[L] = best_L
        if best_L and L >= 8:
            scale_record(W, best_L["state"], nf, device, [9, 12, 15, 18], 100,
                         out_dir / "progressive", f"prog_shell_L{L}_screen")
    return champs


def scale_record(W, state, nf, device, ns, B, out_dir, tag, use_sealed=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    L = int(state["L"])
    snap = state.get("snap")
    if snap is None:
        snap = state["phi"]
    sc = scale1000_snap(
        W, state["gamma"].float(), snap.float(), L, nf, device, ns,
        snap_mode=state.get("snap_mode", "shell"),
        snap_shared=bool(state.get("snap_shared", False)),
        fourier_M=int(state.get("fourier_M", 2)),
        max_count=B, use_sealed=use_sealed,
    )
    rows = []
    for n in ns:
        m = sc.get(n)
        if not m:
            continue
        ref = SEALED_REF[n]
        rows.append({
            "tag": tag, "n": n, "L": L,
            "Med": m["median_inv_p"], "B90": m["B90_inv_p"],
            "Med_x": m["median_inv_p"] / ref["Med"],
            "B90_x": m["B90_inv_p"] / ref["B90"],
            "snap_mode": state.get("snap_mode", "shell"),
            "snap_shared": state.get("snap_shared", False),
            "B": m["B"],
        })
    (out_dir / f"scale1000_{tag}.json").write_text(
        json.dumps(_jsonable({"tag": tag, "scale": sc, "rows": rows}), indent=2)
    )
    return sc, rows


def run_snap_bakeoff(W, nf, device, *, L_list, seeds, steps, max_train, max_val, out_dir,
                     modes=None, screen_B=100, final_B=1000):
    """Train each SNAP head (perL + shared) at each L; screen then full scale champs."""
    modes = modes or list(SNAP_MODES)
    bake_dir = out_dir / "snap_bakeoff"
    bake_dir.mkdir(parents=True, exist_ok=True)
    head_champs = {}  # tag -> best state at max L

    for mode in modes:
        for shared in (False, True):
            if stopped():
                return head_champs
            tag = snap_tag(mode, shared)
            log(f"==== SNAP bakeoff {tag} ====")
            prev = {s: (None, None) for s in seeds}
            best_mode = None
            for L in L_list:
                if stopped():
                    break
                best_L = None
                for seed in seeds:
                    g0, s0 = prev[seed]
                    b = train_one(
                        W, L, nf, device, max_train=max_train, max_val=max_val,
                        seed=seed, lr=2e-3, steps=steps,
                        gamma0=g0, snap0=s0,
                        snap_mode=mode, snap_shared=shared,
                        arm_tag=f"prog_{tag}", out_dir=bake_dir / tag,
                    )
                    st = b["state"]
                    prev[seed] = (st["gamma"], st["snap"])
                    append_leaderboard(
                        out_dir / "leaderboard.csv",
                        lb_row(f"prog_{tag}", L, seed, 2e-3, steps, "log_inv", 0.1,
                               b, mode, shared),
                    )
                    if best_L is None or b["score"] < best_L["score"]:
                        best_L = b
                if best_L and L == L_list[-1]:
                    best_mode = best_L
                    # screen
                    scale_record(W, best_L["state"], nf, device, [9, 12, 15, 18],
                                 screen_B, bake_dir / tag, f"{tag}_L{L}_screen")
            if best_mode:
                head_champs[tag] = best_mode
                # store screen as provisional scale for ranking
                sc_path = bake_dir / tag / f"scale1000_{tag}_L{L_list[-1]}_screen.json"
                if sc_path.exists():
                    import json as _json
                    blob = _json.loads(sc_path.read_text())
                    # rebuild dict keyed by int n
                    sc = {}
                    for row in blob.get("rows", []):
                        sc[int(row["n"])] = {
                            "median_inv_p": row["Med"], "B90_inv_p": row["B90"], "B": row["B"]
                        }
                    best_mode["scale_screen"] = sc
                    n9 = sc.get(9, {})
                    log(f"HEAD {tag} screen n9 Med={n9.get('median_inv_p')} "
                        f"B90={n9.get('B90_inv_p')}")

    # Full B=final_B only for top-4 heads by screen n9 Med
    def _n9(b):
        sc = b.get("scale_screen") or {}
        return sc.get(9, {}).get("median_inv_p", 1e9)
    ranked = sorted(head_champs.items(), key=lambda kv: _n9(kv[1]))
    for tag, best_mode in ranked[:4]:
        if stopped():
            break
        log(f"=== full scale1000 B={final_B} for {tag} ===")
        sc, rows = scale_record(
            W, best_mode["state"], nf, device, [9, 12, 15, 18], final_B,
            bake_dir / tag, f"{tag}_L{L_list[-1]}_B{final_B}",
        )
        best_mode["scale1000"] = sc
        n9 = sc.get(9, {})
        log(f"HEAD {tag} FULL n9 Med={n9.get('median_inv_p')} B90={n9.get('B90_inv_p')}")
    # for non-top, use screen as scale1000 proxy in table
    for tag, best_mode in head_champs.items():
        if "scale1000" not in best_mode and "scale_screen" in best_mode:
            best_mode["scale1000"] = best_mode["scale_screen"]
            best_mode["scale1000_is_screen"] = True
    return head_champs


def write_snap_results(head_champs: dict, path: Path):
    lines = [
        "# SNAP head bakeoff — RESULTS",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "Fixed max_train=200 n∈{6,9}. Motif Nov1st-final.",
        "",
        "## Sealed reference",
        "",
        "| n | Med | B90 |",
        "|---|-----|-----|",
    ]
    for n in (9, 12, 15, 18):
        r = SEALED_REF[n]
        lines.append(f"| {n} | {r['Med']:.3f} | {r['B90']:.3f} |")
    lines += ["", "## Per SNAP head (scale1000)", "",
              "| head | n9 Med/B90 | n12 | n15 | n18 | n9 Med× |",
              "|------|------------|-----|-----|-----|---------|"]
    ranked = []
    for tag, b in head_champs.items():
        sc = b.get("scale1000") or {}
        def fmt(n):
            m = sc.get(n)
            return f"{m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}" if m else "—"
        n9 = sc.get(9)
        medx = (n9["median_inv_p"] / SEALED_REF[9]["Med"]) if n9 else 999
        ranked.append((medx, tag, sc))
        lines.append(
            f"| `{tag}` | {fmt(9)} | {fmt(12)} | {fmt(15)} | {fmt(18)} | {medx:.3f}× |"
        )
    ranked.sort()
    lines += ["", "## Closest to sealed (by n9 Med×)", ""]
    if ranked:
        medx, tag, sc = ranked[0]
        lines.append(f"**Winner:** `{tag}` (n9 Med×sealed = {medx:.3f})")
        lines.append("")
        for n in (9, 12, 15, 18):
            m = sc.get(n)
            if not m:
                continue
            ref = SEALED_REF[n]
            lines.append(
                f"- n{n}: Med {m['median_inv_p']:.3f} ({m['median_inv_p']/ref['Med']:.3f}×) "
                f"B90 {m['B90_inv_p']:.3f} ({m['B90_inv_p']/ref['B90']:.3f}×)"
            )
    lines += ["", "Leaderboard: `../leaderboard.csv`.", ""]
    path.write_text("\n".join(lines))


def write_results_md(summary, path: Path):
    lines = [
        "# Nov1st-final angle + SNAP explore — RESULTS",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        summary.get("notes", ""),
        "",
        "See also `snap_bakeoff/RESULTS.md`.",
        "",
    ]
    path.write_text("\n".join(lines))


def load_warm_shell_from_disk(out_dir: Path, seeds, prefer_L=16):
    """Resume progressive from saved params_L{prefer_L}."""
    warm = {}
    prog = out_dir / "progressive"
    if not prog.exists():
        return None
    for seed in seeds:
        # find highest L <= prefer_L or exact
        cands = sorted(prog.glob(f"params_prog_*_L*_seed{seed}.pt"))
        # also old naming
        cands += sorted(prog.glob(f"params_prog_log_inv_lp0.1_L*_seed{seed}.pt"))
        best = None
        best_L = -1
        for pt in cands:
            try:
                st = torch.load(pt, map_location="cpu", weights_only=False)
                L = int(st.get("L", 0))
                if L > best_L and L <= prefer_L + 20:  # allow any
                    # prefer closest to prefer_L if going up
                    if L >= prefer_L or best is None:
                        snap = st.get("snap")
                        if snap is None:
                            snap = st.get("phi")
                        if snap is None:
                            continue
                        best = (st["gamma"], snap, L)
                        best_L = L
            except Exception:
                pass
        # pick max L for each seed
        best = None
        best_L = -1
        for pt in set(cands):
            try:
                st = torch.load(pt, map_location="cpu", weights_only=False)
                L = int(st.get("L", 0))
                snap = st.get("snap", st.get("phi"))
                if snap is None:
                    continue
                if L > best_L:
                    best = (st["gamma"], snap, L)
                    best_L = L
            except Exception:
                pass
        if best:
            warm[seed] = best
            log(f"disk warm seed{seed} L{best[2]}")
    return warm or None


def beats_sealed(sc: dict) -> tuple[bool, dict]:
    """Strict dual-beat vs sealed Med AND B90 at every n in SEALED_REF."""
    detail = {}
    ok = True
    for n, ref in SEALED_REF.items():
        m = sc.get(n)
        if not m:
            ok = False
            detail[n] = {"missing": True}
            continue
        med_ok = m["median_inv_p"] < ref["Med"]
        b90_ok = m["B90_inv_p"] < ref["B90"]
        detail[n] = {
            "Med": m["median_inv_p"], "B90": m["B90_inv_p"],
            "Med_x": m["median_inv_p"] / ref["Med"],
            "B90_x": m["B90_inv_p"] / ref["B90"],
            "med_beat": med_ok, "b90_beat": b90_ok,
            "dual_beat": med_ok and b90_ok,
        }
        if not (med_ok and b90_ok):
            ok = False
    return ok, detail


def _load_pt_state(pt: Path) -> dict | None:
    if not pt.exists():
        return None
    st = torch.load(pt, map_location="cpu", weights_only=False)
    if "snap" not in st and st.get("phi") is not None:
        st["snap"] = st["phi"]
    return st


def collect_polish_starts(out_dir: Path) -> list[dict]:
    """Warm starts for wave-2 polish (do not wipe prior results)."""
    starts = []
    prog = out_dir / "progressive"
    for seed in (0, 1, 2):
        pt = prog / f"params_prog_snap_shell_perL_L32_seed{seed}.pt"
        st = _load_pt_state(pt)
        if st:
            starts.append({"name": f"shell_L32_seed{seed}", "state": st, "path": str(pt)})
    # bakeoff free / shell / bits champs at L32
    bake = out_dir / "snap_bakeoff"
    for tag in ("snap_free_perL", "snap_shell_perL", "snap_bits_perL", "snap_x_pop_perL"):
        d = bake / tag
        if not d.exists():
            continue
        cands = sorted(d.glob("params_*_L32_seed*.pt"))
        for pt in cands:
            st = _load_pt_state(pt)
            if st:
                starts.append({"name": f"{tag}_{pt.stem}", "state": st, "path": str(pt)})
    # dedupe by path
    seen = set()
    uniq = []
    for s in starts:
        if s["path"] in seen:
            continue
        seen.add(s["path"])
        uniq.append(s)
    return uniq


def run_polish_wave(W, nf, device, *, max_train, max_val, out_dir, scale_B=1000):
    """Wave-2: keep max_train=200; polish best angles harder (steps/LBFGS/loss/lr)."""
    polish_dir = out_dir / "polish"
    polish_dir.mkdir(parents=True, exist_ok=True)
    starts = collect_polish_starts(out_dir)
    log(f"polish starts raw: {len(starts)}")
    # Prioritize: progressive shell seeds, then free/bits bakeoff L32 seed0 only
    def rank(s):
        n = s["name"]
        if n == "shell_L32_seed0": return (0, 0)
        if n.startswith("shell_L32_seed"): return (0, int(n.split("seed")[-1]))
        if "free_perL" in n and "seed0" in n: return (1, 0)
        if "free_perL" in n and "seed1" in n: return (1, 1)
        if "bits_perL" in n and "seed0" in n: return (2, 0)
        if "shell_perL" in n and "seed0" in n and "bake" not in n: return (3, 0)
        return (9, 0)
    starts = sorted(starts, key=rank)
    # Cap starts to top 5 to keep nonstop moving
    starts = [s for s in starts if rank(s)[0] <= 2][:5]
    log(f"polish starts selected: {[s['name'] for s in starts]}")
    # Tier-A recipes on all starts; tier-B only if n9 Med_x <= 1.05 after tier-A best
    recipes_A = [
        ("adam800_lbfgs8", 800, 2e-3, "log_inv", 0.1, 4, 12, 0),
        ("cvar_adam600_lbfgs", 600, 2e-3, "cvar", 0.1, 4, 10, 30),
        ("adam600_lr1e3_lbfgs", 600, 1e-3, "log_inv", 0.1, 4, 10, 10),
    ]
    recipes_B = [
        ("adam600_lr5e3_lbfgs", 600, 5e-3, "log_inv", 0.1, 3, 8, 20),
        ("cvar_q09_adam600", 600, 2e-3, "cvar", 0.05, 3, 8, 40),
        ("raw_inv_adam400_lbfgs", 400, 2e-3, "raw_inv", 0.1, 4, 10, 50),
        ("adam1200_lbfgs12", 1200, 2e-3, "log_inv", 0.1, 5, 14, 60),
    ]
    results = []
    beat_any = False

    for s in starts:
        if stopped():
            break
        st0 = s["state"]
        mode = st0.get("snap_mode", "shell")
        shared = bool(st0.get("snap_shared", False))
        L = int(st0.get("L", 32))
        g0 = st0["gamma"]
        sn0 = st0.get("snap")
        if sn0 is None:
            sn0 = st0.get("phi")
        best_n9x = 9e9
        recipes = list(recipes_A)
        tier_b_added = False
        for rname, steps, lr, loss, leakpen, lb_r, lb_i, soff in recipes:
            if stopped():
                break
            seed = int(st0.get("seed", 0)) + soff
            arm = f"polish_{mode}_{'sh' if shared else 'pL'}_{rname}"
            log(f"=== polish {s['name']} → {arm} L={L} seed={seed} ===")
            best = train_one(
                W, L, nf, device, max_train=max_train, max_val=max_val, seed=seed,
                lr=lr, steps=steps, gamma0=g0, snap0=sn0,
                snap_mode=mode, snap_shared=shared,
                loss_kind=loss, leakpen=leakpen,
                lbfgs_mid=True, lbfgs_final=True,
                lbfgs_rounds=lb_r, lbfgs_iter=lb_i,
                arm_tag=arm, out_dir=polish_dir / mode,
            )
            append_leaderboard(
                out_dir / "leaderboard.csv",
                lb_row(arm, L, seed, lr, steps, loss, leakpen, best, mode, shared),
            )
            # full scale on every polish (goal is beat sealed)
            sc, rows = scale_record(
                W, best["state"], nf, device, [9, 12, 15, 18], scale_B,
                polish_dir, f"{arm}_seed{seed}",
            )
            beat, detail = beats_sealed(sc)
            n9 = detail.get(9, {})
            entry = {
                "start": s["name"], "arm": arm, "seed": seed, "L": L,
                "mode": mode, "shared": shared, "loss": loss, "lr": lr, "steps": steps,
                "path": best.get("path"), "beat_sealed": beat, "detail": detail,
                "n9_Med": n9.get("Med"), "n9_B90": n9.get("B90"),
                "n9_Med_x": n9.get("Med_x"), "n9_B90_x": n9.get("B90_x"),
                "val_n9_Med": best["val"][9]["median_inv_p"] if best.get("val") else None,
            }
            results.append(entry)
            (polish_dir / f"meta_{arm}_seed{seed}.json").write_text(
                json.dumps(_jsonable(entry), indent=2)
            )
            if beat:
                beat_any = True
                log(f"*** BEAT SEALED *** {arm} seed{seed} n9={n9.get('Med'):.4f}/{n9.get('B90'):.4f}")
                # write ping file for parent
                (out_dir / "BEAT_SEALED.json").write_text(
                    json.dumps(_jsonable(entry), indent=2)
                )
            else:
                # closeness log
                xs = " ".join(
                    f"n{n}:{detail[n].get('Med_x', float('nan')):.3f}/"
                    f"{detail[n].get('B90_x', float('nan')):.3f}"
                    for n in (9, 12, 15, 18) if n in detail and not detail[n].get("missing")
                )
                log(f"no-beat {arm} seed{seed} ×sealed {xs}")
            if n9.get("Med_x") is not None:
                best_n9x = min(best_n9x, float(n9["Med_x"]))
            # After tier-A done for this start, queue tier-B if close
            # After finishing current tier-A list (before extend), unlock tier-B if close
            if (not tier_b_added) and rname == recipes_A[-1][0]:
                if best_n9x <= 1.05:
                    log(f"tier-B unlock for {s['name']} best_n9x={best_n9x:.4f}")
                    recipes.extend(recipes_B)
                    tier_b_added = True

    # write polish RESULTS
    lines = [
        "# Polish wave — RESULTS",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "Fixed max_train=200. Goal: dual-beat sealed Med AND B90 at n9/12/15/18.",
        "",
        "## Sealed bar",
        "",
        "| n | Med | B90 |",
        "|---|-----|-----|",
    ]
    for n, r in SEALED_REF.items():
        lines.append(f"| {n} | {r['Med']} | {r['B90']} |")
    lines += ["", "## Runs", "",
              "| arm | start | n9 Med/B90 | n9× | n12× | n15× | n18× | beat? |",
              "|-----|-------|------------|-----|------|------|------|-------|"]
    for e in results:
        d = e["detail"]
        def fx(n):
            if n not in d or d[n].get("missing"):
                return "—"
            return f"{d[n]['Med_x']:.3f}/{d[n]['B90_x']:.3f}"
        n9m = e.get("n9_Med"); n9b = e.get("n9_B90")
        lines.append(
            f"| `{e['arm']}` | {e['start'][:40]} | "
            f"{n9m:.3f}/{n9b:.3f} | {fx(9)} | {fx(12)} | {fx(15)} | {fx(18)} | "
            f"{'YES' if e['beat_sealed'] else 'no'} |"
        )
    # closest by n9 Med×
    valid = [e for e in results if e.get("n9_Med_x") is not None]
    if valid:
        best = min(valid, key=lambda e: (e["n9_Med_x"], e.get("n9_B90_x") or 9e9))
        lines += ["", "## Closest (n9 Med×)", "",
                  f"**{best['arm']}** from `{best['start']}` — "
                  f"n9 {best['n9_Med']:.3f}/{best['n9_B90']:.3f} "
                  f"({best['n9_Med_x']:.3f}× / {best['n9_B90_x']:.3f}×)",
                  "",
                  f"Any sealed dual-beat: **{'YES' if beat_any else 'no'}**."]
    (polish_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (polish_dir / "summary.json").write_text(json.dumps(_jsonable({
        "n_runs": len(results), "beat_any": beat_any, "results": results,
    }), indent=2))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--phase", default="all",
                    help="all|progressive|snap_bakeoff|scale_champ|resume_shell|polish")
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_train", type=int, default=200)
    ap.add_argument("--max_val", type=int, default=40)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--snap_L", default="8,16,32", help="L ladder for snap bakeoff")
    ap.add_argument("--start_L", type=int, default=1)
    ap.add_argument("--modes", default=",".join(SNAP_MODES))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    modes = [x for x in args.modes.split(",") if x.strip()]
    snap_Ls = tuple(int(x) for x in args.snap_L.split(",") if x.strip())
    summary = {"phase": args.phase, "max_train": args.max_train, "heads": describe_heads()}

    sm = smoke_overlap(n=6, L=2, nf=args.nf, device=args.device, seed=0)
    summary["smoke"] = sm
    (OUT / "smoke.json").write_text(json.dumps(_jsonable(sm), indent=2))
    if not sm["pass"]:
        raise SystemExit("smoke FAIL")
    if args.smoke_only:
        log("smoke-only done"); return

    champs = {}
    if args.phase in ("all", "progressive", "resume_shell"):
        warm = load_warm_shell_from_disk(OUT, seeds)
        start_L = args.start_L
        if warm and args.phase in ("resume_shell", "all"):
            max_saved = max(v[2] for v in warm.values())
            nxt = [L for L in L_LADDER if L > max_saved]
            start_L = nxt[0] if nxt else max_saved
            log(f"resume_shell start_L={start_L} (max_saved={max_saved})")
        champs = run_progressive_shell(
            W, args.nf, device, seeds=seeds, lr=args.lr, steps=args.steps,
            max_train=args.max_train, max_val=args.max_val, out_dir=OUT,
            start_L=start_L, warm_from=warm,
        )

    head_champs = {}
    if args.phase in ("all", "snap_bakeoff"):
        head_champs = run_snap_bakeoff(
            W, args.nf, device, L_list=snap_Ls, seeds=seeds[:2],  # 2 seeds for speed
            steps=args.steps, max_train=args.max_train, max_val=args.max_val,
            out_dir=OUT, modes=modes, screen_B=100, final_B=args.scale_B,
        )
        write_snap_results(head_champs, OUT / "snap_bakeoff" / "RESULTS.md")
        (OUT / "snap_bakeoff" / "summary.json").write_text(
            json.dumps(_jsonable({
                "heads": {k: {"score": v["score"], "path": v.get("path"),
                              "scale": v.get("scale1000")} for k, v in head_champs.items()}
            }), indent=2)
        )

    # scale best shell progressive L32 if present
    if champs.get(32) and args.phase in ("all", "progressive", "resume_shell", "scale_champ"):
        scale_record(W, champs[32]["state"], args.nf, device, [9, 12, 15, 18],
                     args.scale_B, OUT, "shell_prog_L32_B1000")
        torch.save(champs[32]["state"], OUT / "champ_shell_params.pt")

    if args.phase in ("polish",):
        polish_results = run_polish_wave(
            W, args.nf, device, max_train=args.max_train, max_val=args.max_val,
            out_dir=OUT, scale_B=args.scale_B,
        )
        summary["polish_n"] = len(polish_results)
        summary["polish_beat"] = any(r.get("beat_sealed") for r in polish_results)
        if summary["polish_beat"]:
            log("*** POLISH WAVE BEAT SEALED ***")

    # pick closest SNAP head
    if head_champs:
        def n9med(b):
            sc = b.get("scale1000") or {}
            m = sc.get(9)
            return m["median_inv_p"] if m else 1e9
        winner = min(head_champs.items(), key=lambda kv: n9med(kv[1]))
        summary["snap_winner"] = winner[0]
        summary["snap_winner_n9"] = n9med(winner[1])
        torch.save(winner[1]["state"], OUT / "champ_snap_params.pt")
        log(f"SNAP winner: {winner[0]} n9 Med={n9med(winner[1]):.3f}")

    summary["notes"] = (
        f"SNAP winner={summary.get('snap_winner')} "
        f"n9Med={summary.get('snap_winner_n9')}. "
        + ("STOPPED. " if stopped() else "Complete. ")
    )
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    write_results_md(summary, OUT / "RESULTS.md")
    log("DONE →", OUT)


if __name__ == "__main__":
    main()

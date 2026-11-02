#!/usr/bin/env python3
"""Finish scale1000 B=1000 for progressive shell L-ladder {1,4,...,32}.

Parallel instance eval (ProcessPool) to finish under CPU contention.
Picks best seed per L by train-val n9 Med. Writes MD+JSON.
Honors Nov/runs/STOP_ANGLE_EXPLORE. Does not touch polish.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

_NOV = Path(__file__).resolve().parents[1]
_ROOT = _NOV.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.binH_vs_grayH import build_W_BinH  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_explore import _jsonable  # noqa: E402
from Nov.scripts.angle_explore import L_LADDER, OUT, log, stopped  # noqa: E402
from Nov.scripts.nov1st_final_train import SCALE_DIR, SEALED_REF, make_sim  # noqa: E402

PROG = OUT / "progressive"
RESULT_MD = OUT / "L_LADDER_scale1000.md"
RESULT_JSON = OUT / "L_LADDER_scale1000.json"
NS = [9, 12, 15, 18]
B = 1000
NF = 8
N_WORKERS = int(os.environ.get("LADDER_WORKERS", "3"))


def pick_best_checkpoint(L: int) -> dict | None:
    metas = sorted(PROG.glob(f"meta_*_L{L}_seed*.json"))
    best = None
    for mf in metas:
        meta = json.loads(mf.read_text())
        seed = int(meta.get("seed", -1))
        val = meta.get("val") or {}
        n9 = (val.get("9") or val.get(9) or {})
        med = float(n9.get("Med", float("inf")))
        pt = None
        if meta.get("path"):
            cand = Path(meta["path"])
            if cand.exists():
                pt = cand
        if pt is None:
            cands = sorted(PROG.glob(f"params_*_L{L}_seed{seed}.pt"))
            shell = [c for c in cands if "shell" in c.name]
            pt = (shell or cands)[0] if (shell or cands) else None
        if pt is None or not pt.exists():
            continue
        row = {
            "L": L,
            "seed": seed,
            "n9_med_val": med,
            "n9_b90_val": float(n9.get("B90", float("nan"))),
            "n6_med_val": float((val.get("6") or val.get(6) or {}).get("Med", float("nan"))),
            "meta_path": str(mf),
            "params_path": str(pt),
            "arm": meta.get("arm") or meta.get("snap_mode") or "",
        }
        if best is None or row["n9_med_val"] < best["n9_med_val"]:
            best = row
    return best


def load_state(pt: Path) -> dict:
    st = torch.load(pt, map_location="cpu", weights_only=False)
    if "snap" not in st and st.get("phi") is not None:
        st["snap"] = st["phi"]
    st.setdefault("snap_mode", "shell")
    st.setdefault("snap_shared", False)
    return st


def _eval_chunk(args):
    """Worker: evaluate a contiguous slice of E rows. Top-level for pickling."""
    (
        n, L, nf, gamma_np, snap_np, snap_mode, snap_shared, fourier_M, few_M,
        E_np, start, end, progress_every,
    ) = args
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    device = torch.device("cpu")
    W = build_W_BinH(nf=nf)
    sim = make_sim(n, W, L, nf, device)
    gamma = torch.as_tensor(gamma_np, dtype=torch.float32, device=device)
    snap = torch.as_tensor(snap_np, dtype=torch.float32, device=device)
    invs, leaks = [], []
    for b in range(start, end):
        out = sim.forward_dispatch(
            torch.as_tensor(E_np[b], dtype=torch.float32, device=device),
            gamma, snap, use_sealed=False,
            snap_mode=snap_mode, snap_shared=snap_shared,
            fourier_M=fourier_M, few_M=few_M,
        )
        invs.append(float(out["inv_p"].detach().cpu()))
        leaks.append(float(out["p_leak"].detach().cpu()))
        if progress_every and ((b - start + 1) % progress_every == 0):
            print(
                f"{time.strftime('%H:%M:%S')} [ladder-worker] n{n} L{L} "
                f"{b + 1}/{end} (chunk {start}:{end}) Med≈{np.median(invs):.3f}",
                flush=True,
            )
    return start, invs, leaks


def scale1000_parallel(state, nf, ns, max_count=1000, n_workers=N_WORKERS):
    L = int(state["L"])
    gamma_np = state["gamma"].detach().cpu().numpy()
    snap = state.get("snap")
    if snap is None:
        snap = state["phi"]
    snap_np = snap.detach().cpu().numpy()
    snap_mode = state.get("snap_mode", "shell")
    snap_shared = bool(state.get("snap_shared", False))
    fourier_M = int(state.get("fourier_M", 2))
    few_M = int(state.get("few_M", 3))
    out = {}
    if not SCALE_DIR.exists():
        log("scale1000 dir missing")
        return out

    for n in ns:
        if stopped():
            break
        pack = load_pack(n, data_dir=SCALE_DIR)
        key = "scale1000" if "scale1000_seeds" in pack else (
            "test" if "test_seeds" in pack else None
        )
        if key is None:
            continue
        _, _, _, _, E = _split_arrays(pack, key, max_count=max_count)
        if E is None or len(E) == 0:
            continue
        E_np = np.asarray(E)
        Btot = len(E_np)
        # chunk by workers
        n_workers_eff = max(1, min(n_workers, Btot))
        edges = np.linspace(0, Btot, n_workers_eff + 1, dtype=int)
        chunks = []
        prog = 100 if n >= 15 else 200
        for i in range(n_workers_eff):
            s, e = int(edges[i]), int(edges[i + 1])
            if e <= s:
                continue
            chunks.append((
                n, L, nf, gamma_np, snap_np, snap_mode, snap_shared, fourier_M, few_M,
                E_np, s, e, prog,
            ))
        t0 = time.time()
        invs = [None] * Btot
        leaks = [None] * Btot
        if n_workers_eff == 1:
            s, inv, leak = _eval_chunk(chunks[0])
            invs[s:s + len(inv)] = inv
            leaks[s:s + len(leak)] = leak
        else:
            with ProcessPoolExecutor(max_workers=n_workers_eff) as ex:
                futs = {ex.submit(_eval_chunk, c): c[10] for c in chunks}  # start idx
                for fut in as_completed(futs):
                    s, inv, leak = fut.result()
                    invs[s:s + len(inv)] = inv
                    leaks[s:s + len(leak)] = leak
        invs_a = np.asarray(invs, dtype=np.float64)
        leaks_a = np.asarray(leaks, dtype=np.float64)
        med = float(np.median(invs_a))
        b90 = float(np.quantile(invs_a, 0.90))
        m = {
            "n": n, "split": key, "B": Btot,
            "median_inv_p": med, "B90_inv_p": b90,
            "mean_leak": float(np.mean(leaks_a)),
            "S": float(np.log(med) + 0.5 * np.log(b90)) if med > 0 and b90 > 0 else float("nan"),
            "L": L, "snap_mode": snap_mode, "snap_shared": snap_shared,
            "wall_s": time.time() - t0, "n_workers": n_workers_eff,
        }
        out[n] = m
        ref = SEALED_REF.get(n)
        bit = ""
        if ref:
            bit = (f"  vs sealed Med {med / ref['Med']:.3f}× "
                   f"B90 {b90 / ref['B90']:.3f}×")
        log(f"scale1000 n{n} L{L} {snap_mode}: Med={med:.3f} B90={b90:.3f} "
            f"B={Btot} workers={n_workers_eff} wall={m['wall_s']:.1f}s{bit}")
    return out


def scale_record_parallel(state, nf, ns, B, out_dir, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    L = int(state["L"])
    sc = scale1000_parallel(state, nf, ns, max_count=B, n_workers=N_WORKERS)
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
            "wall_s": m.get("wall_s"),
            "n_workers": m.get("n_workers"),
        })
    (out_dir / f"scale1000_{tag}.json").write_text(
        json.dumps(_jsonable({"tag": tag, "scale": sc, "rows": rows}), indent=2)
    )
    return sc, rows


def write_outputs(entries: list[dict]):
    payload = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "B": B,
        "ns": NS,
        "sealed": SEALED_REF,
        "L_list": list(L_LADDER),
        "n_workers": N_WORKERS,
        "entries": entries,
    }
    RESULT_JSON.write_text(json.dumps(_jsonable(payload), indent=2))

    lines = [
        "# Progressive shell L-ladder — scale1000 B=1000",
        "",
        f"_Updated {time.strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai_",
        "",
        "Best progressive checkpoint per L (lowest train-val n9 Med). "
        "Motif Nov1st-final, SNAP head = shell (perL). "
        f"Eval B={B} on n∈{NS} (parallel workers={N_WORKERS}).",
        "",
        "## Sealed bar (L=32)",
        "",
        "| n | Med | B90 |",
        "|---|-----|-----|",
    ]
    for n in NS:
        r = SEALED_REF[n]
        lines.append(f"| {n} | {r['Med']:.3f} | {r['B90']:.3f} |")

    lines += [
        "",
        "## Ladder (Med / B90) and ×sealed",
        "",
        "| L | seed | n9 Med/B90 | n12 Med/B90 | n15 Med/B90 | n18 Med/B90 | "
        "Med× / B90× (n9,12,15,18) |",
        "|---|------|------------|-------------|-------------|-------------|"
        "---------------------------|",
    ]
    for e in entries:
        sc = e.get("scale") or {}
        abs_cells = []
        x_parts = []
        for n in NS:
            m = sc.get(str(n)) or sc.get(n) or {}
            if not m:
                abs_cells.append("—")
                x_parts.append("—")
                continue
            med = m["median_inv_p"]
            b90 = m["B90_inv_p"]
            ref = SEALED_REF[n]
            abs_cells.append(f"{med:.3f}/{b90:.3f}")
            x_parts.append(f"{med / ref['Med']:.3f}/{b90 / ref['B90']:.3f}")
        lines.append(
            f"| {e['L']} | {e['seed']} | "
            + " | ".join(abs_cells)
            + " | "
            + ", ".join(x_parts)
            + " |"
        )

    lines += [
        "",
        "## Checkpoint selection",
        "",
        "| L | seed | train-val n9 Med | params |",
        "|---|------|------------------|--------|",
    ]
    for e in entries:
        lines.append(
            f"| {e['L']} | {e['seed']} | {e['n9_med_val']:.4f} | "
            f"`{Path(e['params_path']).name}` |"
        )
    lines += ["", f"JSON: `{RESULT_JSON.name}`", ""]
    RESULT_MD.write_text("\n".join(lines))
    log(f"wrote {RESULT_MD} and {RESULT_JSON}")


def main():
    # Keep parent process lean; workers pin to 1 thread each.
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    PROG.mkdir(parents=True, exist_ok=True)

    picks = {}
    for L in L_LADDER:
        p = pick_best_checkpoint(L)
        if p is None:
            log(f"MISSING checkpoint for L={L}")
        else:
            log(
                f"pick L={L} seed{p['seed']} n9_val={p['n9_med_val']:.4f} "
                f"{Path(p['params_path']).name}"
            )
        picks[L] = p

    missing = [L for L, p in picks.items() if p is None]
    if missing:
        from Nov.scripts.angle_explore import load_warm_shell_from_disk, train_one

        device = torch.device("cpu")
        W = build_W_BinH(nf=NF)
        log(f"training missing L={missing}")
        warm = load_warm_shell_from_disk(
            OUT, [0, 1, 2], prefer_L=min(missing) - 1 if min(missing) > 1 else 1
        )
        prev = {s: (None, None) for s in (0, 1, 2)}
        if warm:
            for s, (g, sn, L0) in warm.items():
                prev[s] = (g, sn)
                log(f"warm seed{s} from L{L0}")
        for L in L_LADDER:
            if L not in missing:
                continue
            if stopped():
                log("STOP file present — abort")
                return
            for seed in (0, 1, 2):
                if stopped():
                    break
                g0, s0 = prev[seed]
                best = train_one(
                    W, L, NF, device, max_train=200, max_val=40, seed=seed,
                    lr=2e-3, steps=400, gamma0=g0, snap0=s0,
                    snap_mode="shell", snap_shared=False,
                    arm_tag="prog_snap_shell_perL", out_dir=PROG,
                )
                st = best["state"]
                prev[seed] = (st["gamma"], st["snap"])
            picks[L] = pick_best_checkpoint(L)

    entries_by_L: dict[int, dict] = {}
    for L in reversed(L_LADDER):
        if stopped():
            log("STOP file present — writing partial results")
            break
        p = picks.get(L)
        if p is None:
            log(f"skip L={L} still missing")
            continue
        st = load_state(Path(p["params_path"]))
        tag = f"prog_shell_L{L}_B1000"
        out_json = PROG / f"scale1000_{tag}.json"
        if out_json.exists():
            try:
                old = json.loads(out_json.read_text())
                rows = old.get("rows") or []
                ok = len(rows) >= 4 and all(int(r.get("B", 0)) >= B for r in rows)
                if ok:
                    log(f"reuse existing {out_json.name}")
                    sc = {int(k): v for k, v in (old.get("scale") or {}).items()}
                    entries_by_L[L] = {**p, "scale": sc, "tag": tag, "reused": True}
                    write_outputs([entries_by_L[x] for x in L_LADDER if x in entries_by_L])
                    continue
            except Exception as e:
                log(f"reuse failed for L={L}: {e}")

        log(f"=== scale1000 B={B} L={L} seed{p['seed']} workers={N_WORKERS} ===")
        t0 = time.time()
        sc, rows = scale_record_parallel(st, NF, NS, B, PROG, tag)
        log(f"done L={L} in {time.time() - t0:.1f}s")
        entries_by_L[L] = {**p, "scale": sc, "tag": tag, "rows": rows, "reused": False}
        write_outputs([entries_by_L[x] for x in L_LADDER if x in entries_by_L])

    entries = [entries_by_L[x] for x in L_LADDER if x in entries_by_L]
    write_outputs(entries)
    log("L-ladder scale1000 complete")


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on some platforms
    main()

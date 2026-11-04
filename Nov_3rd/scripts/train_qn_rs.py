#!/usr/bin/env python3
"""Train QN-RS on {6,9} and eval fresh CORE against matched Fourier QAOA.

Phase A freezes (u, v) at the Fourier-QAOA modes and trains (s, w, p).
Phase B unfreezes every coefficient. s = w = 0 is Fourier QAOA, so the
warm start is the baseline the smoke has to beat.

Loss: mean log(inv / inv_FourierQAOA), weights w6=1, w9=2, 20% n9 holdout.
No penalty on r2, r3 — shrinkage is the kill the gate is watching for.
p is clamped to [-1, 2], the same box as power_p.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCR = Path(__file__).resolve().parent
if str(_SCR) not in sys.path:
    sys.path.insert(0, str(_SCR))


def main():
    ap = argparse.ArgumentParser(description="QN-RS train / CORE eval / gate0")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--Q", type=int, default=6)
    ap.add_argument("--Qp", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--fourier-params", type=str, required=True)
    ap.add_argument("--fourier-eval", type=str, default="", help="published Fourier QAOA eval.json")
    ap.add_argument("--packs", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--adam-steps", type=int, default=600)
    ap.add_argument("--adam-lr", type=float, default=1e-3)
    ap.add_argument("--lbfgs-rounds", type=int, default=12)
    ap.add_argument("--lbfgs-iter", type=int, default=8)
    ap.add_argument("--batch-per-n", type=int, default=32)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--w6", type=float, default=1.0)
    ap.add_argument("--w9", type=float, default=2.0)
    ap.add_argument("--eval-B", type=int, default=1000)
    ap.add_argument("--eval-ns", type=str, default="6,9,12,15")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--gate0-tol", type=float, default=1e-3)
    ap.add_argument("--gate0-only", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)

    from Nov_3rd import settings as S
    from Nov_3rd.ansatz import make_sim
    from Nov_3rd.data_io import load_split_E, log
    from Nov_3rd.qn_rs import (
        P_HI,
        P_LO,
        load_fourier_modes,
        paired_median_ci,
        param_count,
        residual_ratio,
        schedules,
    )
    from explore_OCT29.lib.qaoa import qaoa_psucc

    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "params.pt"
    Q = int(args.Q)
    Qp = int(args.Qp)
    L = int(args.L)
    if Q > L or Qp > L:
        raise SystemExit(f"Q={Q} Q'={Qp} exceed L={L}")
    eval_ns = [int(x) for x in args.eval_ns.split(",") if x.strip()]
    weights = {6: float(args.w6), 9: float(args.w9)}
    published = {}
    if args.fourier_eval:
        blob = json.loads(Path(args.fourier_eval).read_text())
        for k, v in blob.get("per_n", {}).items():
            published[int(k)] = float(v["Med"])

    u0, v0 = load_fourier_modes(args.fourier_params, Q)
    log(f"QN-RS L={L} Q={Q} Q'={Qp}: Fourier modes from {args.fourier_params}")

    def load_train_n(n: int) -> torch.Tensor:
        d = np.load(Path(args.packs) / f"train_n{n}.npz", allow_pickle=True)
        E = np.asarray(d["train_E"])
        if args.max_train is not None:
            E = E[: int(args.max_train)]
        return torch.as_tensor(E, dtype=torch.float32)

    def qaoa_inv(betas, gammas, E: torch.Tensor, n: int) -> torch.Tensor:
        invs = []
        chunk = 64 if n <= 9 else (32 if n <= 12 else 16)
        for s0 in range(0, len(E), chunk):
            Eb = E[s0 : s0 + chunk].to(device)
            ps = qaoa_psucc(betas, gammas, Eb, n)
            invs.append((1.0 / ps.clamp(min=1e-12)).detach())
        return torch.cat(invs)

    def eval_batch_size(n: int) -> int:
        if n <= 6:
            return 64
        if n <= 9:
            return 32
        if n <= 12:
            return 8
        return 4

    @torch.no_grad()
    def eval_core(u, v, s, w, p) -> dict:
        """QN-RS inv versus the frozen Fourier-QAOA angles (u0, v0), not the trained ones.

        Phase B may move u and v. The smoke bar is the matched baseline those
        coefficients were initialized to.
        """
        rows = {}
        inv_blob = {}
        base = schedules(u0.to(device), v0.to(device), torch.zeros(Qp, device=device), torch.zeros(Qp, device=device), 0.0, L, 6)
        for n in eval_ns:
            sch = schedules(u, v, s, w, p, L, n)
            E, key = load_split_E(n, "scale1000", args.eval_B, device, data_dir=S.SCALE1000_DIR)
            sim = make_sim(n, L, device)
            bs = eval_batch_size(n)
            inv_parts = []
            leak_parts = []
            t0 = time.time()
            B = len(E)
            for s0 in range(0, B, bs):
                out = sim.forward_nov3rd_batch(E[s0 : s0 + bs], sch["gamma"], sch["phi"])
                inv_parts.append(out["inv_p"].detach().cpu())
                leak_parts.append(out["p_leak"].detach().cpu())
            inv = torch.cat(inv_parts).numpy()
            leak = torch.cat(leak_parts).numpy()
            fq = qaoa_inv(base["beta"], base["gamma"], E.cpu(), n).detach().cpu().numpy()
            boot_seed = 10_000 + int(args.seed) * 17 + int(n)
            boot = paired_median_ci(inv, fq, n_boot=args.boot, seed=boot_seed)
            med = float(np.median(inv))
            fq_med = float(np.median(fq))
            rows[str(n)] = {
                "Med": med,
                "B90": float(np.quantile(inv, 0.90)),
                "Fourier_Med": fq_med,
                "published_Fourier_Med": published.get(n),
                "abs_med_vs_published": None if n not in published else abs(med - published[n]),
                "median_abs_instance_diff": float(np.median(np.abs(inv - fq))),
                "max_abs_instance_diff": float(np.max(np.abs(inv - fq))),
                "delta_med": boot["delta_med"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "ci_excludes_0": boot["excludes_0"],
                "ours_better": boot["ours_better"],
                "B": int(inv.shape[0]),
                "mean_leak": float(leak.mean()),
                "wall_s": time.time() - t0,
                "split": key,
            }
            inv_blob[f"n{n}_qnrs"] = inv.astype(np.float64)
            inv_blob[f"n{n}_fourier"] = fq.astype(np.float64)
            pub = published.get(n)
            log(
                f"QN-RS L{L} s{args.seed} n{n}: Med={med:.4f} FQ={fq_med:.4f} "
                f"Δ={boot['delta_med']:+.4f} CI=[{boot['ci_lo']:+.4f},{boot['ci_hi']:+.4f}] "
                f"pub={pub} wall={rows[str(n)]['wall_s']:.1f}s"
            )
        sch15 = schedules(u, v, s, w, p, L, 15)
        ratio = residual_ratio(sch15["beta"], sch15["r2"], sch15["r3"], sch15["a"])
        return rows, inv_blob, ratio, sch15

    if args.gate0_only:
        rows, inv_blob, ratio, sch15 = eval_core(u0.to(device), v0.to(device), torch.zeros(Qp), torch.zeros(Qp), 0.0)
        n15 = rows["15"]
        published_15 = float(published.get(15, n15["Fourier_Med"]))
        ok_pub = abs(n15["Med"] - published_15) <= args.gate0_tol
        ok_pair = abs(n15["Med"] - n15["Fourier_Med"]) <= args.gate0_tol
        # s=w=0 must also match instance-wise, not only the median.
        ok_inst = n15["median_abs_instance_diff"] <= args.gate0_tol
        payload = {
            "gate0_pass": bool(ok_pub and ok_pair and ok_inst),
            "tol": args.gate0_tol,
            "published_n15": published_15,
            "ok_vs_published": ok_pub,
            "ok_vs_paired_fourier": ok_pair,
            "ok_instance": ok_inst,
            "residual_ratio_n15": ratio,
            "p": 0.0,
            "per_n": rows,
            "L": L,
            "Q": Q,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        }
        (out_dir / "gate0.json").write_text(json.dumps(payload, indent=2) + "\n")
        np.savez_compressed(out_dir / "gate0_inv.npz", **inv_blob)
        log(
            f"GATE0 L{L}: {'PASS' if payload['gate0_pass'] else 'FAIL'} "
            f"n15 Med={n15['Med']:.6f} pub={published_15:.6f} "
            f"|Δpub|={abs(n15['Med'] - published_15):.3e} "
            f"med|inst|={n15['median_abs_instance_diff']:.3e}"
        )
        return 0 if payload["gate0_pass"] else 2

    if not args.skip_train:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        u = u0.to(device).float()
        v = v0.to(device).float()
        s = torch.nn.Parameter(torch.zeros(Qp, device=device))
        w = torch.nn.Parameter(torch.zeros(Qp, device=device))
        p = torch.nn.Parameter(torch.zeros((), device=device))

        bundles = {}
        with torch.no_grad():
            sch0 = schedules(u, v, s, w, p, L, 6)
        for n in (6, 9):
            E = load_train_n(n).to(device)
            sim = make_sim(n, L, device)
            # Reference is this depth's Fourier QAOA, not the free angle vector.
            inv_q = qaoa_inv(sch0["beta"], sch0["gamma"], E.cpu(), n).to(device)
            bundles[n] = (E, inv_q, sim)
            log(f"QN-RS L{L} s{args.seed}: train n{n} B={len(E)} FQ_Med={float(inv_q.median()):.4f}")

        n9_B = len(bundles[9][0])
        perm = rng.permutation(n9_B)
        n_hold = max(1, int(round(args.holdout_frac * n9_B)))
        hold_idx = perm[:n_hold]
        train_pool = perm[n_hold:]

        def active_params():
            ps = [s, w, p]
            if isinstance(u, torch.nn.Parameter):
                ps = [u, v] + ps
            return ps

        def log_ratios(n: int, idx) -> torch.Tensor:
            E, inv_q, sim = bundles[n]
            sch = schedules(u, v, s, w, p, L, n)
            idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=E.device)
            inv = sim.forward_nov3rd_batch(E.index_select(0, idx_t), sch["gamma"], sch["phi"])["inv_p"]
            ref = inv_q.index_select(0, idx_t).clamp(min=1e-12)
            return torch.log((inv / ref).clamp(min=1e-12))

        def batch_loss(idx_map) -> torch.Tensor:
            terms = []
            wsum = 0.0
            for n in (6, 9):
                logs = log_ratios(n, idx_map[n])
                wt = float(weights[n])
                terms.append(wt * logs.mean())
                wsum += wt
            return torch.stack(terms).sum() / max(wsum, 1e-12)

        def sample_idx():
            idx_map = {}
            for n in (6, 9):
                pool = train_pool if n == 9 else np.arange(len(bundles[n][0]))
                take = min(args.batch_per_n, len(pool))
                idx_map[n] = rng.choice(pool, size=take, replace=False)
            return idx_map

        @torch.no_grad()
        def holdout() -> float:
            logs = log_ratios(9, hold_idx)
            return float(logs.median().detach())

        def snapshot():
            return {
                "u": u.detach().cpu().clone(),
                "v": v.detach().cpu().clone(),
                "s": s.detach().cpu().clone(),
                "w": w.detach().cpu().clone(),
                "p": float(p.detach().cpu()),
            }

        def restore(snap):
            with torch.no_grad():
                if isinstance(u, torch.nn.Parameter):
                    u.copy_(snap["u"].to(device))
                    v.copy_(snap["v"].to(device))
                else:
                    u.copy_(snap["u"].to(device))
                    v.copy_(snap["v"].to(device))
                s.copy_(snap["s"].to(device))
                w.copy_(snap["w"].to(device))
                p.fill_(snap["p"])

        def clamp_p():
            with torch.no_grad():
                p.clamp_(P_LO, P_HI)

        hist = []
        t0 = time.time()
        best = {"score": float("inf"), "snap": snapshot(), "step": "init"}
        sc0 = holdout()
        best["score"] = sc0
        log(f"QN-RS L{L} s{args.seed}: init holdout med logR={sc0:.5f} params={param_count(Q, Qp)}")

        def run_phase(tag: str, steps: int, rounds: int):
            opt = torch.optim.Adam(active_params(), lr=args.adam_lr)
            for step in range(steps):
                idx_map = sample_idx()
                opt.zero_grad()
                loss = batch_loss(idx_map)
                loss.backward()
                gnorm = float(torch.nn.utils.clip_grad_norm_(active_params(), 5.0))
                opt.step()
                clamp_p()
                hist.append({"phase": tag, "step": step, "loss": float(loss.detach()), "gnorm": gnorm})
                if step % 25 == 0 or step == steps - 1:
                    sc = holdout()
                    mark = ""
                    if sc < best["score"]:
                        best["score"] = sc
                        best["snap"] = snapshot()
                        best["step"] = f"{tag}-adam{step}"
                        mark = " BEST"
                    with torch.no_grad():
                        sch = schedules(u, v, s, w, p, L, 15)
                        ratio = residual_ratio(sch["beta"], sch["r2"], sch["r3"], sch["a"])
                    log(
                        f"QN-RS L{L} s{args.seed} {tag} adam {step}/{steps} "
                        f"loss={hist[-1]['loss']:.5f} hold={sc:.5f} p={float(p.detach()):.3f} "
                        f"ratio15={ratio:.3f} |s|={float(s.detach().norm()):.3f} g={gnorm:.3f}{mark}"
                    )
            for r in range(rounds):
                opt_l = torch.optim.LBFGS(
                    active_params(),
                    lr=0.25,
                    max_iter=args.lbfgs_iter,
                    history_size=16,
                    line_search_fn="strong_wolfe",
                )
                idx_map = sample_idx()

                def closure():
                    opt_l.zero_grad()
                    lt = batch_loss(idx_map)
                    lt.backward()
                    torch.nn.utils.clip_grad_norm_(active_params(), 5.0)
                    return lt

                try:
                    loss = opt_l.step(closure)
                    loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
                except Exception as e:
                    log(f"QN-RS L{L} s{args.seed} {tag} LBFGS r{r} skip: {e}")
                    continue
                clamp_p()
                hist.append({"phase": tag, "round": r, "loss": loss_f})
                sc = holdout()
                mark = ""
                if sc < best["score"]:
                    best["score"] = sc
                    best["snap"] = snapshot()
                    best["step"] = f"{tag}-lbfgs{r}"
                    mark = " BEST"
                log(f"QN-RS L{L} s{args.seed} {tag} lbfgs {r}/{rounds} loss={loss_f:.5f} hold={sc:.5f}{mark}")

        log(f"QN-RS L{L} s{args.seed}: Phase A freeze u,v train s,w,p ({2 * Qp + 1} params)")
        run_phase("A", args.adam_steps, args.lbfgs_rounds)
        if best["snap"] is not None:
            log(f"QN-RS L{L} s{args.seed}: Phase B starts from best holdout {best['score']:.5f} @ {best['step']}")
            restore(best["snap"])
        u = torch.nn.Parameter(u.detach().clone())
        v = torch.nn.Parameter(v.detach().clone())
        log(f"QN-RS L{L} s{args.seed}: Phase B unfreeze all ({param_count(Q, Qp)} params)")
        run_phase("B", args.adam_steps, args.lbfgs_rounds)
        if best["snap"] is not None and best["score"] == best["score"]:
            log(f"QN-RS L{L} s{args.seed}: restore best holdout {best['score']:.5f} @ {best['step']}")
            restore(best["snap"])

        with torch.no_grad():
            sch = schedules(u, v, s, w, p, L, 15)
            ratio = residual_ratio(sch["beta"], sch["r2"], sch["r3"], sch["a"])
        state = {
            "idea": "qn_rs",
            "L": L,
            "Q": Q,
            "Qp": Qp,
            "seed": args.seed,
            "u": u.detach().cpu(),
            "v": v.detach().cpu(),
            "s": s.detach().cpu(),
            "w": w.detach().cpu(),
            "p": float(p.detach().cpu()),
            "n_params": param_count(Q, Qp),
            "residual_ratio_n15": ratio,
            "holdout_best": best["score"],
            "holdout_best_step": best["step"],
            "init_from": str(args.fourier_params),
            "wall_s": time.time() - t0,
            "hist_tail": hist[-40:],
            "formula": "phi=-beta*k + a(n)*(r2*k^2+r3*k^3); a=(8/(n-1))^p; gamma,beta Fourier",
        }
        torch.save(state, params_path)
        meta = {
            "done": True,
            "idea": "qn_rs",
            "L": L,
            "Q": Q,
            "Qp": Qp,
            "seed": args.seed,
            "n_params": param_count(Q, Qp),
            "p": state["p"],
            "residual_ratio_n15": ratio,
            "s_l2": float(state["s"].norm()),
            "w_l2": float(state["w"].norm()),
            "holdout_best": best["score"],
            "holdout_best_step": best["step"],
            "wall_s": state["wall_s"],
            "init_from": state["init_from"],
            "final_loss": hist[-1].get("loss") if hist else None,
            "params": str(params_path),
        }
        (out_dir / "train.json").write_text(json.dumps(meta, indent=2) + "\n")
        log(
            f"QN-RS L{L} s{args.seed} saved {params_path} wall={state['wall_s']:.0f}s "
            f"p={state['p']:.4f} ratio15={ratio:.4f} @ {best['step']}"
        )
    elif not params_path.exists():
        raise SystemExit(f"--skip-train but no {params_path}")

    if args.skip_eval:
        return 0

    state = torch.load(params_path, map_location="cpu", weights_only=False)
    rows, inv_blob, ratio, sch15 = eval_core(
        state["u"].to(device),
        state["v"].to(device),
        state["s"].to(device),
        state["w"].to(device),
        float(state["p"]),
    )
    payload = {
        "idea": "qn_rs",
        "L": L,
        "Q": Q,
        "Qp": Qp,
        "seed": args.seed,
        "eval_B": args.eval_B,
        "p": float(state["p"]),
        "residual_ratio_n15": ratio,
        "a15": float(sch15["a"].detach().cpu()) if torch.is_tensor(sch15["a"]) else float(sch15["a"]),
        "s_l2": float(state["s"].norm()),
        "w_l2": float(state["w"].norm()),
        "holdout_best": state.get("holdout_best"),
        "holdout_best_step": state.get("holdout_best_step"),
        "init_from": state.get("init_from"),
        "per_n": rows,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (out_dir / "eval.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(out_dir / "inv.npz", **inv_blob)
    lines = [
        f"# QN-RS L={L} seed={args.seed}",
        "",
        f"p={state['p']:.4f} residual_ratio_n15={ratio:.4f} checkpoint={state.get('holdout_best_step')}",
        "",
        "| n | QN-RS Med | Fourier Med | Δ | 95% CI | better |",
        "|---|-----------|-------------|---|--------|--------|",
    ]
    for n in eval_ns:
        r = rows[str(n)]
        lines.append(
            f"| {n} | {r['Med']:.4f} | {r['Fourier_Med']:.4f} | {r['delta_med']:+.4f} | "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['ours_better']} |"
        )
    lines.append("")
    (out_dir / "eval.md").write_text("\n".join(lines) + "\n")
    log(f"QN-RS wrote {out_dir / 'eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

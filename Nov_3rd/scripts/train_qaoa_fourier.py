#!/usr/bin/env python3
"""Fourier QAOA with the same modes and INTERP rule as FM-WS.

    β_ℓ = Σ_{q=1..Q} B_q v_q(ℓ)     v = DCT-IV cosine
    γ_ℓ = Σ_{q=1..Q} A_q u_q(ℓ)     u = DST-IV sine

Warm start: DCT/DST of the free L=16 QAOA angles, truncated to Q.
INTERP: keep (A, B) and rebuild the basis at the new depth (pad q>Q with 0).

Loss matches the fresh250 QAOA trainer (equal n6/n9 mean −log p_succ, batch
32+32) plus the same Adam+LBFGS round count as FM-WS, so a Med gap is not an
optimizer-budget artifact. Learning rate stays on the QAOA scale (0.05).
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


def main():
    ap = argparse.ArgumentParser(description="Fourier QAOA train / INTERP / CORE eval")
    ap.add_argument("--Q", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--angles", type=str, required=True, help="free QAOA angles.json to DCT/DST")
    ap.add_argument("--init-ckpt", type=str, default="", help="INTERP: previous Fourier QAOA params.pt")
    ap.add_argument("--packs", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--adam-steps", type=int, default=600)
    ap.add_argument("--adam-lr", type=float, default=0.05)
    ap.add_argument("--lbfgs-rounds", type=int, default=12)
    ap.add_argument("--lbfgs-iter", type=int, default=8)
    ap.add_argument("--batch-per-n", type=int, default=32)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--eval-B", type=int, default=1000)
    ap.add_argument("--eval-ns", type=str, default="6,9,12,15")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)

    from Nov_3rd.data_io import log
    from Nov_3rd.ensure_qaoa import eval_scale_ladder
    from Nov_3rd.fm_ws import qaoa_angles_from_modes, qaoa_modes_from_angles
    from explore_OCT29.lib.qaoa import qaoa_psucc

    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "params.pt"
    Q = int(args.Q)
    L = int(args.L)
    eval_ns = [int(x) for x in args.eval_ns.split(",") if x.strip()]

    def load_pack(n: int) -> torch.Tensor:
        d = np.load(Path(args.packs) / f"train_n{n}.npz", allow_pickle=True)
        E = np.asarray(d["train_E"])
        if args.max_train is not None:
            E = E[: int(args.max_train)]
        return torch.as_tensor(E, dtype=torch.float32)

    def load_init():
        if args.init_ckpt:
            prev = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
            if int(prev["Q"]) != Q:
                raise SystemExit(f"init ckpt Q={prev['Q']} != --Q {Q}")
            log(f"Fourier QAOA INTERP init from {args.init_ckpt} (was L={prev.get('basis_L')})")
            return prev["A"].float(), prev["B"].float(), "interp", str(args.init_ckpt)
        blob = json.loads(Path(args.angles).read_text())
        best = blob["best"]
        betas = torch.tensor(best["betas"], dtype=torch.float32)
        gammas = torch.tensor(best["gammas"], dtype=torch.float32)
        modes = qaoa_modes_from_angles(betas, gammas, Q)
        log(
            f"Fourier QAOA warm DCT/DST of {args.angles} L={int(betas.numel())} → Q={Q}"
        )
        return modes["A"], modes["B"], "free_qaoa_dct", str(args.angles)

    if not args.skip_train:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        A0, B0, how, src = load_init()
        A = torch.nn.Parameter(A0.to(device).float())
        B = torch.nn.Parameter(B0.to(device).float())
        E6 = load_pack(6)
        E9 = load_pack(9)
        log(f"Fourier QAOA Q{Q} L{L}: n6={len(E6)} n9={len(E9)} steps={args.adam_steps} lr={args.adam_lr}")

        def loss_on(idx6, idx9):
            betas, gammas = qaoa_angles_from_modes(A, B, L)
            ps6 = qaoa_psucc(betas, gammas, E6[idx6].to(device), 6)
            ps9 = qaoa_psucc(betas, gammas, E9[idx9].to(device), 9)
            loss6 = -torch.log(ps6.clamp_min(1e-12)).mean()
            loss9 = -torch.log(ps9.clamp_min(1e-12)).mean()
            return 0.5 * (loss6 + loss9)

        opt = torch.optim.Adam([A, B], lr=args.adam_lr)
        hist = []
        t0 = time.time()
        for step in range(args.adam_steps):
            i6 = rng.choice(len(E6), size=min(args.batch_per_n, len(E6)), replace=False)
            i9 = rng.choice(len(E9), size=min(args.batch_per_n, len(E9)), replace=False)
            opt.zero_grad()
            loss = loss_on(i6, i9)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([A, B], 5.0)
            opt.step()
            hist.append(float(loss.detach().cpu()))
            if step % 25 == 0 or step == args.adam_steps - 1:
                log(
                    f"Fourier QAOA Q{Q} L{L} adam {step}/{args.adam_steps} "
                    f"loss={hist[-1]:.4f} wall={time.time()-t0:.0f}s"
                )

        for r in range(args.lbfgs_rounds):
            opt_l = torch.optim.LBFGS(
                [A, B], lr=0.25, max_iter=args.lbfgs_iter, history_size=16, line_search_fn="strong_wolfe"
            )
            i6 = rng.choice(len(E6), size=min(args.batch_per_n, len(E6)), replace=False)
            i9 = rng.choice(len(E9), size=min(args.batch_per_n, len(E9)), replace=False)

            def closure():
                opt_l.zero_grad()
                lt = loss_on(i6, i9)
                lt.backward()
                torch.nn.utils.clip_grad_norm_([A, B], 5.0)
                return lt

            try:
                loss = opt_l.step(closure)
                loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
            except Exception as e:
                log(f"Fourier QAOA Q{Q} L{L} LBFGS r{r} skip: {e}")
                continue
            hist.append(loss_f)
            log(f"Fourier QAOA Q{Q} L{L} lbfgs {r}/{args.lbfgs_rounds} loss={loss_f:.4f}")

        with torch.no_grad():
            betas, gammas = qaoa_angles_from_modes(A, B, L)
        state = {
            "idea": "qaoa_fourier",
            "Q": Q,
            "L": L,
            "basis_L": L,
            "seed": args.seed,
            "A": A.detach().cpu(),
            "B": B.detach().cpu(),
            "betas": betas.detach().cpu(),
            "gammas": gammas.detach().cpu(),
            "init": how,
            "init_from": src,
            "n_params": int(2 * Q),
            "wall_s": time.time() - t0,
            "final_loss": hist[-1] if hist else None,
            "hist_tail": hist[-30:],
        }
        torch.save(state, params_path)
        # angles.json so load_qaoa() can read the stretched schedule
        angles_blob = {
            "L": L,
            "Q": Q,
            "idea": "qaoa_fourier",
            "best": {
                "loss": state["final_loss"],
                "betas": betas.detach().cpu().tolist(),
                "gammas": gammas.detach().cpu().tolist(),
            },
            "init": how,
            "init_from": src,
            "wall_s": state["wall_s"],
        }
        (out_dir / "angles.json").write_text(json.dumps(angles_blob, indent=2) + "\n")
        (out_dir / "train.json").write_text(
            json.dumps(
                {
                    "done": True,
                    "idea": "qaoa_fourier",
                    "Q": Q,
                    "L": L,
                    "seed": args.seed,
                    "n_params": 2 * Q,
                    "final_loss": state["final_loss"],
                    "wall_s": state["wall_s"],
                    "init": how,
                    "init_from": src,
                    "params": str(params_path),
                },
                indent=2,
            )
            + "\n"
        )
        log(f"Fourier QAOA Q{Q} L{L} saved {params_path} wall={state['wall_s']:.0f}s loss={state['final_loss']}")
    elif not params_path.exists():
        raise SystemExit(f"--skip-train but no {params_path}")

    if args.skip_eval:
        return 0

    state = torch.load(params_path, map_location="cpu", weights_only=False)
    # Always rebuild at --L so an INTERP eval uses the padded basis, not a stale angle vector.
    betas, gammas = qaoa_angles_from_modes(state["A"].to(device), state["B"].to(device), L)
    log(f"Fourier QAOA Q{Q} L{L}: eval scale1000 ns={eval_ns} B={args.eval_B}")
    ladder = eval_scale_ladder(L, betas, gammas, device=device, ns=eval_ns, max_count=args.eval_B)
    payload = {
        "idea": "qaoa_fourier",
        "Q": Q,
        "L": L,
        "seed": args.seed,
        "eval_B": args.eval_B,
        "final_loss": state.get("final_loss"),
        "per_n": {
            str(n): {
                "Med": float(row["median_inv_p"]),
                "B90": float(row["B90_inv_p"]),
                "B": int(row["B"]),
                "wall_s": float(row.get("wall_s", float("nan"))),
            }
            for n, row in ladder.items()
        },
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (out_dir / "eval.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        f"# Fourier QAOA Q={Q} L={L}",
        "",
        "| n | Med | B90 | B |",
        "|---|-----|-----|---|",
    ]
    for n in eval_ns:
        r = payload["per_n"][str(n)]
        lines.append(f"| {n} | {r['Med']:.4f} | {r['B90']:.4f} | {r['B']} |")
    lines.append("")
    (out_dir / "eval.md").write_text("\n".join(lines) + "\n")
    log(f"Fourier QAOA wrote {out_dir / 'eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

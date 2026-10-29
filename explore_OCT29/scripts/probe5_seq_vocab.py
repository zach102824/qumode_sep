#!/usr/bin/env python3
"""Opus probe 5 — seq-evolve with W/SNAP0/PROJ atoms; promote on fitted b.

Train {n9,n12}; eval n9/12/15; fit b_Med; discard if worse than projector
fixed-angle baseline (probe2) or Fourier L12 ceiling reference.

Stop: touch runs/STOP_PROBE5
Forever unless --max_trials N.
"""
from __future__ import annotations

import argparse
import csv
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
from explore_OCT29.lib.seq_w_ansatz import (  # noqa: E402
    SEED_W_MOTIFS,
    WMotif,
    MultiCavityWSeqAnsatz,
    init_w_params,
    load_W,
    mutate_w_motif,
)
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402

OUT = _PKG / "runs" / "probe5_seq_vocab"
STOP = _PKG / "runs" / "STOP_PROBE5"
P2_PARAMS = _PKG / "runs" / "probe2_train912" / "projector_912_params.pt"
P3_JSON = _PKG / "runs" / "probe3_fourier" / "results.json"
P3_DEEP = _PKG / "runs" / "probe3_fourier_deep" / "results.json"
P3_DEEP_MODES = _PKG / "runs" / "probe3_fourier_deep" / "modes_q3_Ltrain6.pt"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[probe5]", *a, flush=True)


def _S(med, b90):
    return 0.5 * (math.log(max(med, 1e-12)) + math.log(max(b90, 1e-12)))


def _loss(out, leakpen=0.1):
    return torch.log(out["inv_p"].clamp(min=1.0)) + leakpen * out["p_leak"]


@torch.no_grad()
def eval_wmotif(n, motif, params, W, nf, max_count, device, split="val"):
    pack = load_pack(n)
    seeds, _, _, _, E = _split_arrays(pack, split, max_count=max_count)
    cfg = AnsatzConfig(n=n, L=motif.L, nf=nf)
    sim = MultiCavityWSeqAnsatz(cfg, motif, W, device=device).to(device)
    p = {k: v.detach().to(device) for k, v in params.items()}
    invs, leaks = [], []
    t0 = time.time()
    for b in range(len(seeds)):
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        out = sim.forward_wmotif(Eb, p)
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
    invs_a = np.asarray(invs)
    med = float(np.median(invs_a))
    b90 = float(np.quantile(invs_a, 0.9))
    return {
        "n": n, "B": len(invs), "median_inv_p": med, "B90_inv_p": b90,
        "S": _S(med, b90), "mean_leak": float(np.mean(leaks)),
        "wall_s": time.time() - t0, "motif": motif.name, "L": motif.L,
    }


def eval_ladder(motif, params, W, nf, device, max_counts):
    out = {}
    for n, mc in max_counts.items():
        if not (_PKG / "data" / f"data_n{n}.npz").exists():
            continue
        out[n] = eval_wmotif(n, motif, params, W, nf, mc, device)
    return out


def fit_b_from_ladder(ladder):
    ns, meds, b90s = [], [], []
    for n in (9, 12, 15):
        m = ladder.get(n)
        if m and "median_inv_p" in m:
            ns.append(n); meds.append(m["median_inv_p"]); b90s.append(m["B90_inv_p"])
    return {"Med": fit_b(ns, meds), "B90": fit_b(ns, b90s)}


def short_train(
    motif, W, steps, max_train, max_val, nf, seed, device,
    warm_g=None, warm_b=None, lr=4e-3,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    params = init_w_params(motif, seed=seed, device=device, warm_gamma=warm_g, warm_beta=warm_b)
    sims, Es = {}, {}
    for n in (9, 12):
        pack = load_pack(n)
        _, _, _, _, E = _split_arrays(pack, "train", max_count=max_train)
        if E is None or len(E) == 0:
            _, _, _, _, E = _split_arrays(pack, "val", max_count=max_train)
        Es[n] = E
        sims[n] = MultiCavityWSeqAnsatz(
            AnsatzConfig(n=n, L=motif.L, nf=nf), motif, W, device=device
        ).to(device)

    opt = torch.optim.Adam(list(params.values()), lr=lr)
    best = {"score": float("inf"), "params": None, "val": None, "step": -1}
    cut = max(steps - 15, steps * 2 // 3)

    for step in range(steps):
        if step == cut and any(p.numel() <= 16 for p in params.values()):
            # LBFGS only when few params (projector-like)
            n_p = sum(p.numel() for p in params.values())
            if n_p <= 24:
                opt = torch.optim.LBFGS(
                    list(params.values()), lr=0.15, max_iter=3,
                    history_size=15, line_search_fn="strong_wolfe",
                )

        def closure():
            opt.zero_grad(set_to_none=True)
            losses = []
            for n, sim in sims.items():
                idx = rng.choice(len(Es[n]), size=min(5, len(Es[n])), replace=False)
                for i in idx:
                    Eb = torch.as_tensor(Es[n][i], dtype=torch.float32, device=device)
                    out = sim.forward_wmotif(Eb, params)
                    losses.append(_loss(out))
            loss = torch.stack(losses).mean()
            loss.backward()
            return loss

        if isinstance(opt, torch.optim.LBFGS):
            try:
                loss = opt.step(closure)
            except Exception:
                loss = closure()
                opt_a = torch.optim.Adam(list(params.values()), lr=lr)
                opt_a.step()
                opt = opt_a
        else:
            loss = closure()
            torch.nn.utils.clip_grad_norm_(list(params.values()), 5.0)
            opt.step()

        if not ((step % 5 == 0) or step < 2 or step + 1 == steps):
            continue
        vals = {
            n: eval_wmotif(n, motif, params, W, nf, max_val, device)
            for n in (9, 12)
        }
        sc = 0.4 * vals[9]["S"] + 0.6 * vals[12]["S"]
        if sc < best["score"]:
            best = {
                "score": sc,
                "params": {k: v.detach().cpu().clone() for k, v in params.items()},
                "val": vals,
                "step": step,
            }
            log(
                f"  step {step}: Scomb={sc:.3f} "
                + " ".join(f"n{n}:{vals[n]['median_inv_p']:.2f}/{vals[n]['B90_inv_p']:.2f}" for n in (9, 12))
            )
    if best["params"] is None:
        best["params"] = {k: v.detach().cpu().clone() for k, v in params.items()}
        best["val"] = {n: eval_wmotif(n, motif, params, W, nf, max_val, device) for n in (9, 12)}
    return best


def load_baselines():
    """Ceilings from Fourier q3_Ltrain6 (deepen winner) + probe2 fallback warms."""
    base = {
        # NEW GATE (Fourier deepen d1399b4d): must beat/match these to promote
        "b_med_gate": 1.24,
        "med15_fourier_L12": 7.70,   # q3_Ltrain6 @ deploy L12
        "med15_fourier_L20": 5.99,   # q3_Ltrain6 @ deploy L20
        "med15_proj_L4": 30.9,       # legacy probe2 (reject bar only)
        "med12_proj": 17.5,
        "fourier_tag": "q3_Ltrain6",
        "u_gamma": None,
        "u_beta": None,
        "warm_gamma": None,
        "warm_beta": None,
    }
    # Prefer deepen q3 modes
    if P3_DEEP.exists():
        js = json.loads(P3_DEEP.read_text())
        for block in js.get("runs", []):
            if block.get("tag") == "q3_Ltrain6":
                base["med15_fourier_L12"] = float(block.get("med15_L12") or base["med15_fourier_L12"])
                base["med15_fourier_L20"] = float(block.get("med15_Lmax") or base["med15_fourier_L20"])
                base["b_med_gate"] = float(block.get("b_Med_Lmax") or block.get("b_Med_L12") or 1.24)
                base["u_gamma"] = block.get("u_gamma")
                base["u_beta"] = block.get("u_beta")
                break
    if P3_DEEP_MODES.exists():
        blob = torch.load(P3_DEEP_MODES, map_location="cpu", weights_only=False)
        base["u_gamma"] = blob["u_gamma"].tolist() if torch.is_tensor(blob["u_gamma"]) else blob["u_gamma"]
        base["u_beta"] = blob["u_beta"].tolist() if torch.is_tensor(blob["u_beta"]) else blob["u_beta"]
    # free-angle warm fallback (probe2) if no Fourier modes
    if P2_PARAMS.exists():
        blob = torch.load(P2_PARAMS, map_location="cpu", weights_only=False)
        base["warm_gamma"] = blob["gamma"].tolist() if torch.is_tensor(blob["gamma"]) else blob["gamma"]
        base["warm_beta"] = blob["beta"].tolist() if torch.is_tensor(blob["beta"]) else blob["beta"]
    return base


def fourier_angles_for_L(baselines: dict, L: int):
    """Sample q3 Fourier modes at depth L → (gamma, beta) length-L lists."""
    ug, ub = baselines.get("u_gamma"), baselines.get("u_beta")
    if ug is None or ub is None:
        return baselines.get("warm_gamma"), baselines.get("warm_beta")
    from explore_OCT29.scripts.probe3_fourier import angles_from_modes
    g, b = angles_from_modes(torch.as_tensor(ug, dtype=torch.float32),
                             torch.as_tensor(ub, dtype=torch.float32), L)
    return g.tolist(), b.tolist()


def write_status(out_dir, champ, trials_done, baselines):
    lines = [
        "# Opus probe 5 — W-atom seq vocab evolve\n\n",
        f"_Updated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_\n\n",
        f"Trials done: {trials_done}\n\n",
        "## Promote gate (Fourier deepen)\n\n",
        f"- **q3_Ltrain6** n15 Med ≲ **{baselines['med15_fourier_L12']:.3g}** @L12 "
        f"/ **{baselines.get('med15_fourier_L20', 5.99):.3g}** @L20\n",
        f"- b_Med gate ≲ **{baselines['b_med_gate']:.3f}**\n",
        f"- Legacy probe2 n15≈{baselines['med15_proj_L4']} (reject-only)\n\n",
        "## Current seq champ (must beat Fourier to matter)\n\n",
    ]
    if champ is None:
        lines.append("_none yet — Fourier q3_Ltrain6 remains fixed-angle champ_\n")
    else:
        lines.append(f"- motif: `{champ['motif']['name']}`\n")
        lines.append(f"- b_Med={champ['b_med']:.4f} b_B90={champ.get('b_b90', float('nan')):.4f}\n")
        lad = champ.get("ladder", {})
        for n in (9, 12, 15):
            m = lad.get(n) or lad.get(str(n)) or {}
            if "median_inv_p" in m:
                lines.append(f"- n{n}: {m['median_inv_p']:.4g}/{m['B90_inv_p']:.4g}\n")
        m15 = lad.get(15) or lad.get("15") or {}
        ceil = baselines['med15_fourier_L12']
        m15v = m15.get('median_inv_p')
        beat = (m15v is not None) and (m15v <= ceil * 1.05)
        lines.append(f"- vs Fourier L12 n15: champ={m15v} ceiling={ceil:.3g} "
                     f"{'BEATS/MATCHES' if beat else 'LOSES'}\n")
    lines.append("\n## Note\n\nPromote only if n15 Med and b are within noise of q3_Ltrain6 "
                 "(≲7.7 Med / b≲1.24). Mediocre projector mutants are rejected.\n")
    (out_dir / "RESULTS.md").write_text("".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_trials", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--confirm_steps", type=int, default=120)
    ap.add_argument("--max_train", type=int, default=32)
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--max_val15", type=int, default=6)
    ap.add_argument("--nf", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps_mutate", type=float, default=0.7)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = load_W(nf=args.nf, device=device)
    baselines = load_baselines()
    log(f"baselines={ {k:v for k,v in baselines.items() if k not in ('warm_gamma','warm_beta')} }")

    max_counts = {9: args.max_val, 12: args.max_val, 15: args.max_val15}
    csv_path = OUT / "trials.csv"
    fieldnames = [
        "trial", "mode", "keep", "motif", "L", "n_params", "b_med", "b_b90",
        "Med9", "B90_9", "Med12", "B90_12", "Med15", "B90_15", "wall_s", "best_b",
    ]
    new = not csv_path.exists()
    rng = np.random.default_rng(args.seed)
    champ = None
    # Gate = Fourier deepen; seq champ must beat this (not legacy probe2)
    best_b = float(baselines["b_med_gate"])
    best_med15 = float(baselines["med15_fourier_L12"])
    parent = SEED_W_MOTIFS[0]
    seed_i = 0
    trial = 0

    champ_path = OUT / "champ.json"
    if champ_path.exists():
        try:
            prev = json.loads(champ_path.read_text())
            prev_med15 = float(prev["ladder"]["15"]["median_inv_p"])
            prev_b = float(prev["b_med"])
            # Only resume if prior champ already meets new Fourier gate (within 15% Med noise)
            if prev_med15 <= best_med15 * 1.15 and prev_b <= best_b * 1.15:
                champ = prev
                best_b = min(best_b, prev_b)
                best_med15 = min(best_med15, prev_med15)
                parent = WMotif.from_dict(champ["motif"])
                log(f"resumed champ {parent.name} b={best_b:.4f} Med15={best_med15:.3g}")
            else:
                archived = OUT / "champ_pre_fourier_gate.json"
                archived.write_text(json.dumps(prev, indent=2) + "\n")
                log(f"archived weak champ Med15={prev_med15:.3g} b={prev_b:.3g} "
                    f"(gate n15≲{best_med15:.3g} b≲{best_b:.3g}) → {archived.name}")
                champ = None
        except Exception as e:
            log(f"champ load failed: {e}")

    write_status(OUT, champ, trial, baselines)

    with open(csv_path, "a", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if new:
            w.writeheader(); fcsv.flush()

        while True:
            if STOP.exists():
                log("STOP_PROBE5 — exit")
                break
            if args.max_trials and trial >= args.max_trials:
                log("max_trials — exit")
                break
            trial += 1
            t0 = time.time()

            if seed_i < len(SEED_W_MOTIFS):
                cand = SEED_W_MOTIFS[seed_i]
                mode = "seed"
                seed_i += 1
            elif rng.random() < args.eps_mutate:
                cand = mutate_w_motif(parent, rng)
                mode = "mutate"
            else:
                cand = SEED_W_MOTIFS[int(rng.integers(0, len(SEED_W_MOTIFS)))]
                L = int(np.clip(cand.L + int(rng.choice([-1, 0, 1, 2])), 3, 10))
                cand = WMotif(name=f"{cand.name}_rL{L}", prep=cand.prep, layer=cand.layer, L=L)
                mode = "reseed"

            log(f"=== trial {trial} {mode} {cand.name} prep={cand.prep} layer={cand.layer} "
                f"L={cand.L} n_params={cand.n_params()}")

            # warm-start γ/β from Fourier q3 schedule sampled at motif.L (projector-like)
            use_warm = any(o in ("PROJ", "SNAP0") for o in cand.layer) and (
                ("W" in cand.prep) or ("FLAT_PREP" in cand.prep)
            )
            if use_warm:
                warm_g, warm_b = fourier_angles_for_L(baselines, cand.L)
            else:
                warm_g = warm_b = None
            try:
                best = short_train(
                    cand, W, args.steps, args.max_train, args.max_val, args.nf,
                    seed=args.seed + trial, device=device,
                    warm_g=warm_g, warm_b=warm_b,
                )
                ladder = eval_ladder(cand, best["params"], W, args.nf, device, max_counts)
                fits = fit_b_from_ladder(ladder)
                b_med = float(fits["Med"]["b"])
                b_b90 = float(fits["B90"]["b"])
            except Exception as e:
                log(f"trial {trial} FAILED: {e}")
                w.writerow({
                    "trial": trial, "mode": mode, "keep": "error", "motif": cand.name,
                    "L": cand.L, "n_params": cand.n_params(), "b_med": "", "b_b90": "",
                    "Med9": "", "B90_9": "", "Med12": "", "B90_12": "", "Med15": "", "B90_15": "",
                    "wall_s": f"{time.time()-t0:.1f}", "best_b": f"{best_b:.4f}",
                })
                fcsv.flush()
                continue

            m9, m12, m15 = ladder.get(9, {}), ladder.get(12, {}), ladder.get(15, {})
            med15 = m15.get("median_inv_p", float("inf"))
            med12 = m12.get("median_inv_p", float("inf"))

            # promote rules vs Fourier q3_Ltrain6 gate:
            # - reject unless Med15 within ~20% of L12 ceiling OR clearly better b
            # - never promote mediocre mutants (Med15 >> 7.7)
            ceil15 = float(baselines["med15_fourier_L12"])
            keep = False
            reason = "no"
            if med12 > 1e4 or med15 > ceil15 * 3.0:
                reason = "reject_explode"
            elif med15 > ceil15 * 1.25 and b_med > baselines["b_med_gate"] * 1.1:
                reason = "reject_below_fourier"
            elif b_med < best_b * 0.98 and med15 <= ceil15 * 1.10:
                keep = True
                reason = "promote_b"
            elif med15 < best_med15 * 0.97 and b_med <= baselines["b_med_gate"] * 1.12:
                keep = True
                reason = "promote_med15"
            elif champ is None and med15 <= ceil15 * 1.10 and b_med <= baselines["b_med_gate"] * 1.12:
                # first motif that matches Fourier gate becomes seq champ
                keep = True
                reason = "promote_match_fourier"

            row = {
                "trial": trial, "mode": mode, "keep": reason, "motif": cand.name,
                "L": cand.L, "n_params": cand.n_params(),
                "b_med": f"{b_med:.4f}", "b_b90": f"{b_b90:.4f}",
                "Med9": f"{m9.get('median_inv_p', float('nan')):.4f}",
                "B90_9": f"{m9.get('B90_inv_p', float('nan')):.4f}",
                "Med12": f"{m12.get('median_inv_p', float('nan')):.4f}",
                "B90_12": f"{m12.get('B90_inv_p', float('nan')):.4f}",
                "Med15": f"{m15.get('median_inv_p', float('nan')):.4f}",
                "B90_15": f"{m15.get('B90_inv_p', float('nan')):.4f}",
                "wall_s": f"{time.time()-t0:.1f}", "best_b": f"{best_b:.4f}",
            }
            w.writerow(row); fcsv.flush()
            log(
                f"trial {trial} {reason} b={b_med:.4f} Med9/12/15="
                f"{m9.get('median_inv_p', float('nan')):.3g}/"
                f"{m12.get('median_inv_p', float('nan')):.3g}/"
                f"{m15.get('median_inv_p', float('nan')):.3g}"
            )

            if keep:
                log(f"CONFIRM {args.confirm_steps} steps...")
                conf = short_train(
                    cand, W, args.confirm_steps, args.max_train, max(args.max_val, 16),
                    args.nf, seed=args.seed + trial + 9999, device=device,
                    warm_g=warm_g, warm_b=warm_b,
                )
                lad_c = eval_ladder(cand, conf["params"], W, args.nf, device, max_counts)
                fits_c = fit_b_from_ladder(lad_c)
                b_c = float(fits_c["Med"]["b"])
                med15_c = lad_c.get(15, {}).get("median_inv_p", float("inf"))
                ceil15 = float(baselines["med15_fourier_L12"])
                ok = (med15_c <= ceil15 * 1.12 and b_c <= baselines["b_med_gate"] * 1.15) or (
                    med15_c < best_med15 * 0.98 and b_c <= baselines["b_med_gate"] * 1.15
                )
                if ok:
                    champ = {
                        "motif": cand.to_dict(),
                        "b_med": b_c,
                        "b_b90": float(fits_c["B90"]["b"]),
                        "ladder": {str(k): v for k, v in lad_c.items()},
                        "trial": trial,
                        "reason": reason,
                        "vs_fourier_L12_med15": baselines["med15_fourier_L12"],
                        "fourier_gate_tag": baselines.get("fourier_tag", "q3_Ltrain6"),
                        "b_med_gate": baselines["b_med_gate"],
                    }
                    best_b = min(best_b, b_c)
                    best_med15 = min(best_med15, med15_c)
                    parent = cand
                    torch.save(conf["params"], OUT / "champ_params.pt")
                    champ_path.write_text(json.dumps(champ, indent=2) + "\n")
                    (OUT / "results.json").write_text(json.dumps({
                        "champ": champ,
                        "baselines": {k: v for k, v in baselines.items() if k not in ("warm_gamma", "warm_beta")},
                        "trials": trial,
                        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }, indent=2) + "\n")
                    write_status(OUT, champ, trial, baselines)
                    log(f"PROMOTED {cand.name} b={b_c:.4f} Med15={med15_c:.3g}")
                else:
                    log(f"confirm failed b={b_c:.4f} Med15={med15_c:.3g}")
                    if b_med < best_b * 1.05:
                        parent = cand
            else:
                if med12 < 500 and b_med < best_b * 1.1:
                    parent = cand

            write_status(OUT, champ, trial, baselines)

    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

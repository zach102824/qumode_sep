#!/usr/bin/env python3
"""NON-DEPTH #3 schedule mixture (+ optional CVaR) and #4 freedom@L44.

Honesty: still Uc=exp(-iγ Hc). No L48+. No blind-cost.

#3 Equal-shot mixture of L44/L40/L36/soft18 schedules:
  p_mix = mean_k p_k; inv = 1/p_mix. Spearman of log inv_p first.
  Also short CVaR_q80 retrain on L44 (higher tail weight).
  Promote: n18 B90 ≤ 0.97×9.645 AND every-n Med ≤ 1.05× champ Med.

#4 Freedom ladder-1 span_w warm from L44:
  Train {9,12,15}, promote if n15 AND n18 Med ≤ 0.98× champ.

Writes runs/gadget_Uc/renames_Hc/nondepth_34/
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
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_explore import (  # noqa: E402
    _S,
    _jsonable,
    build_W_BinH,
    make_sim,
)
from explore_OCT29.scripts.gadget_Uc_freedom import (  # noqa: E402
    eval_free,
    train_free,
)
from explore_OCT29.scripts.gadget_Uc_soft_n15 import load_train_E  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_tail_retrain import loss_fn  # noqa: E402

OUT = _PKG / "runs" / "gadget_Uc" / "renames_Hc" / "nondepth_34"
SCALE = _PKG / "data" / "scale1000"
CHAMP_PT = _PKG / "runs" / "gadget_Uc" / "unsat_params.pt"

CHAMP = {
    9: {"Med": 1.051, "B90": 1.473},
    12: {"Med": 1.113, "B90": 2.263},
    15: {"Med": 1.233, "B90": 4.243},
    18: {"Med": 1.409, "B90": 9.645},
}
# promote gates
B90_N18_THR = CHAMP[18]["B90"] * 0.97  # ≥3% improvement
MED_MULT = 1.05
FREE_MED_MULT = 0.98  # ≥2% better

MEMBERS = {
    "L44": _PKG / "runs" / "gadget_Uc" / "unsat_params.pt",
    "L40": _PKG / "runs" / "gadget_Uc" / "renames_Hc" / "L40_pad" / "promoted_params.pt",
    "L36": _PKG / "runs" / "gadget_Uc" / "renames_Hc" / "L36_pad" / "promoted_params.pt",
    "soft18": _PKG / "runs" / "gadget_Uc" / "renames_Hc" / "soft_n18_from_champ" / "promoted_params.pt",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[nd34]", *a, flush=True)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if den < 1e-12:
        return 1.0
    return float((ra * rb).sum() / den)


@torch.no_grad()
def inv_array(n, W, gamma, phi, L, nf, device, max_count, progress_every=100):
    pack = load_pack(n, data_dir=SCALE)
    key = "scale1000" if "scale1000_seeds" in pack else "test"
    seeds, _, _, _, E = _split_arrays(pack, key, max_count=max_count)
    sim = make_sim(n, W, L, nf, device)
    invs = []
    t0 = time.time()
    B = len(seeds)
    for b in range(B):
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        out = sim.forward_gadget(Eb, gamma, phi, "unsat")
        invs.append(float(out["inv_p"].cpu()))
        if progress_every and ((b + 1) % progress_every == 0 or b + 1 == B):
            rate = (b + 1) / max(time.time() - t0, 1e-9)
            log(
                f"  n{n} L{L} {b+1}/{B} Med≈{np.median(invs):.3f} "
                f"rate={rate:.2f}/s"
            )
    return np.asarray(invs, dtype=np.float64)


def stats(invs: np.ndarray) -> dict:
    return {
        "B": int(len(invs)),
        "median_inv_p": float(np.median(invs)),
        "B90_inv_p": float(np.quantile(invs, 0.9)),
        "S": _S(float(np.median(invs)), float(np.quantile(invs, 0.9))),
        "mean_inv_p": float(np.mean(invs)),
    }


def load_member(path: Path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    g = st["gamma"].float()
    p = st["phi"].float()
    L = int(st.get("L", g.shape[0]))
    return g, p, L, st


def run_mixture(W, nf, device, ns, B_by_n, members=None):
    members = members or MEMBERS
    per = {}  # name -> n -> inv array
    meta = {}
    for name, path in members.items():
        if not path.exists():
            log(f"SKIP missing {name} {path}")
            continue
        g, p, L, st = load_member(path)
        meta[name] = {"L": L, "path": str(path)}
        per[name] = {}
        log(f"=== member {name} L={L} ===")
        for n in ns:
            B = B_by_n.get(n, 1000)
            invs = inv_array(n, W, g, p, L, nf, device, B)
            per[name][n] = invs
            s = stats(invs)
            log(f"{name} n{n}: Med/B90={s['median_inv_p']:.3f}/{s['B90_inv_p']:.3f}")

    # Spearman on n18 (or largest n present)
    n_sp = 18 if 18 in ns else max(ns)
    names = list(per.keys())
    corr = {}
    if n_sp in next(iter(per.values())):
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                ia = np.log(per[a][n_sp].clip(1e-12))
                ib = np.log(per[b][n_sp].clip(1e-12))
                m = min(len(ia), len(ib))
                rho = spearman(ia[:m], ib[:m])
                corr[f"{a}__{b}"] = rho
                log(f"Spearman loginv n{n_sp} {a} vs {b}: {rho:.4f}")
    max_corr = max(corr.values()) if corr else None
    kill_corr = max_corr is not None and max_corr > 0.95

    # equal-shot mixtures: all, and leave-one-out / top pairs
    mixes = {}
    mix_defs = {
        "all4": names,
        "L44_L40_L36": [x for x in ("L44", "L40", "L36") if x in per],
        "L44_L40": [x for x in ("L44", "L40") if x in per],
        "L44_soft18": [x for x in ("L44", "soft18") if x in per],
        "L44_L40_soft18": [x for x in ("L44", "L40", "soft18") if x in per],
    }
    for mname, mlist in mix_defs.items():
        if len(mlist) < 2:
            continue
        mixes[mname] = {"members": mlist, "ladder": {}}
        for n in ns:
            arrs = [per[nm][n] for nm in mlist]
            mlen = min(len(a) for a in arrs)
            ps = [1.0 / a[:mlen].clip(1e-12) for a in arrs]
            p_mix = np.mean(np.stack(ps, axis=0), axis=0)
            inv_mix = 1.0 / p_mix.clip(1e-12)
            s = stats(inv_mix)
            mixes[mname]["ladder"][n] = s
            log(
                f"mix {mname} n{n}: Med/B90={s['median_inv_p']:.3f}/{s['B90_inv_p']:.3f}"
            )

    # promote check on each mix
    for mname, blk in mixes.items():
        lad = blk["ladder"]
        ok_b90 = lad.get(18, {}).get("B90_inv_p", 1e9) <= B90_N18_THR
        ok_med = all(
            lad[n]["median_inv_p"] <= CHAMP[n]["Med"] * MED_MULT
            for n in ns
            if n in lad
        )
        blk["promote"] = bool(ok_b90 and ok_med)
        blk["gates"] = {
            "n18_B90": lad.get(18, {}).get("B90_inv_p"),
            "thr_B90": B90_N18_THR,
            "ok_b90": ok_b90,
            "ok_med": ok_med,
        }
        log(f"mix {mname} promote={blk['promote']} gates={blk['gates']}")

    # member solo stats for table
    solo = {
        nm: {n: stats(per[nm][n]) for n in ns}
        for nm in names
    }
    return {
        "meta": meta,
        "spearman_n": n_sp,
        "spearman": corr,
        "max_spearman": max_corr,
        "kill_corr_gt_0.95": kill_corr,
        "solo": solo,
        "mixes": mixes,
        "B_by_n": B_by_n,
        "note": "equal-shot p_mix=mean p_k; still renames_Hc",
    }


def train_cvar_L44(W, nf, device, steps=80, batch=6, max_train=40, seed=7, lr=1.2e-3):
    """Higher-tail-weight retrain on sealed L44 (CVaR q80)."""
    st0 = torch.load(CHAMP_PT, map_location="cpu", weights_only=False)
    L = int(st0["L"])
    gamma = torch.nn.Parameter(st0["gamma"].float().clone())
    phi = torch.nn.Parameter(st0["phi"].float().clone())
    rng = np.random.default_rng(seed)
    packs = {}
    for n in (9, 12, 15):
        E, src = load_train_E(n, max_train, device)
        packs[n] = E
        log(f"cvar train n{n} src={src} N={len(E)}")
    sims = {n: make_sim(n, W, L, nf, device) for n in (9, 12, 15)}
    opt = torch.optim.Adam([gamma, phi], lr=lr)
    best = {"score": float("inf"), "state": None, "step": -1}
    for step in range(steps):
        opt.zero_grad()
        invs, leaks = [], []
        for n, E in packs.items():
            # upsample n15
            reps = 2 if n == 15 else 1
            for _ in range(reps):
                idx = int(rng.integers(0, len(E)))
                out = sims[n].forward_gadget(E[idx], gamma, phi, "unsat")
                invs.append(out["inv_p"])
                leaks.append(out["p_leak"])
        loss = loss_fn(torch.stack(invs), "cvar_q80", torch.stack(leaks))
        loss.backward()
        opt.step()
        # cheap val: mean loginv on first 8 of each
        with torch.no_grad():
            vs = []
            for n, E in packs.items():
                for j in range(min(8, len(E))):
                    o = sims[n].forward_gadget(E[j], gamma, phi, "unsat")
                    vs.append(float(torch.log(o["inv_p"].clamp(min=1.0))))
            sc = float(np.mean(vs))
        if sc < best["score"]:
            best = {
                "score": sc,
                "step": step,
                "state": {
                    "gamma": gamma.detach().cpu().clone(),
                    "phi": phi.detach().cpu().clone(),
                    "L": L,
                    "mode": "shell",
                    "cost_mode": "unsat",
                    "W": "BinH",
                    "loss_kind": "cvar_q80",
                    "warm_from": "L44",
                },
            }
        if step % 10 == 0 or step == steps - 1:
            log(f"cvar step {step}: loss={float(loss):.4f} val={sc:.4f} best@{best['step']}")
    return best


def eval_schedule(W, nf, device, gamma, phi, L, ns, B_by_n):
    out = {}
    for n in ns:
        invs = inv_array(n, W, gamma, phi, L, nf, device, B_by_n.get(n, 1000))
        out[n] = stats(invs)
        log(f"eval L{L} n{n}: {out[n]['median_inv_p']:.3f}/{out[n]['B90_inv_p']:.3f}")
    return out


def promote_mix_style(ladder, ns):
    ok_b90 = ladder.get(18, {}).get("B90_inv_p", 1e9) <= B90_N18_THR
    ok_med = all(
        ladder[n]["median_inv_p"] <= CHAMP[n]["Med"] * MED_MULT for n in ns if n in ladder
    )
    return bool(ok_b90 and ok_med), {"ok_b90": ok_b90, "ok_med": ok_med, "thr_B90": B90_N18_THR}


def run_freedom(W, nf, device, steps=60, scale_B_by_n=None):
    scale_B_by_n = scale_B_by_n or {9: 1000, 12: 1000, 15: 1000, 18: 1000}
    st0 = torch.load(CHAMP_PT, map_location="cpu", weights_only=False)
    L = int(st0["L"])
    assert L == 44
    mode = "span_w"
    log(f"=== freedom {mode} L={L} warm L44 train{{9,12,15}} steps={steps} ===")
    best = train_free(
        mode,
        W,
        L,
        nf,
        device,
        steps=steps,
        seed=11,
        gamma0=st0["gamma"].float(),
        phi0=st0["phi"].float(),
        train_ns=(9, 12, 15),
        max_train=40,
        max_val=16,
        lr=1.5e-3,
    )
    st = best["state"]
    torch.save(st, OUT / f"freedom_{mode}_params.pt")
    ladder = {}
    for n in (9, 12, 15, 18):
        B = scale_B_by_n.get(n, 1000)
        log(f"freedom scale n{n} B={B}")
        m = eval_free(
            n,
            W,
            st["gamma"],
            st["phi"],
            L,
            nf,
            device,
            mode,
            w_span=st.get("w_span"),
            split="scale1000",
            max_count=B,
            data_dir=SCALE,
            progress_every=100 if n >= 15 else 200,
        )
        ladder[n] = m
        log(
            f"freedom n{n}: Med/B90={m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}"
        )
    ok15 = ladder[15]["median_inv_p"] <= CHAMP[15]["Med"] * FREE_MED_MULT
    ok18 = ladder[18]["median_inv_p"] <= CHAMP[18]["Med"] * FREE_MED_MULT
    promote = bool(ok15 and ok18)
    return {
        "mode": mode,
        "L": L,
        "train_best_step": best["step"],
        "train_score": best["score"],
        "w_span_raw": None if st.get("w_span") is None else st["w_span"].tolist(),
        "w_span_softplus": None
        if st.get("w_span") is None
        else torch.nn.functional.softplus(st["w_span"]).tolist(),
        "ladder": {str(k): v for k, v in ladder.items()},
        "promote": promote,
        "gates": {
            "thr15": CHAMP[15]["Med"] * FREE_MED_MULT,
            "thr18": CHAMP[18]["Med"] * FREE_MED_MULT,
            "ok15": ok15,
            "ok18": ok18,
        },
    }


def write_md(results: dict):
    lines = [
        "# NON-DEPTH #3+#4 — mixture / CVaR / freedom@L44",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "Honesty: still `Uc=exp(-iγ Hc)`. No L48+. Blind banned. Champ L44 sealed.",
        "",
        f"Champ Med/B90: "
        + ", ".join(f"n{n} {CHAMP[n]['Med']:.3f}/{CHAMP[n]['B90']:.3f}" for n in (9, 12, 15, 18)),
        "",
        "## #3a Equal-shot schedule mixture",
        "",
    ]
    mix = results.get("mixture") or {}
    if mix.get("kill_corr_gt_0.95"):
        lines.append(
            f"**Spearman kill:** max corr={mix.get('max_spearman'):.4f} > 0.95 → "
            "tail largely shared across members (paper sentence)."
        )
    else:
        lines.append(
            f"Max Spearman(log inv_p) n{mix.get('spearman_n')} = "
            f"**{(mix.get('max_spearman') or 0):.4f}** (kill if >0.95)."
        )
    lines += ["", "### Solo members", "", "| member | n9 | n12 | n15 | n18 |", "|--------|----|-----|-----|-----|"]
    for nm, lad in (mix.get("solo") or {}).items():
        cells = []
        for n in (9, 12, 15, 18):
            s = lad.get(n) or lad.get(str(n))
            cells.append(f"{s['median_inv_p']:.3f}/{s['B90_inv_p']:.3f}" if s else "—")
        lines.append(f"| {nm} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### Mixtures (equal-shot)",
        "",
        f"Promote gate: n18 B90 ≤ **{B90_N18_THR:.3f}** (≥3% vs 9.645) AND Med ≤ {MED_MULT}× champ.",
        "",
        "| mix | n9 | n12 | n15 | n18 | promote |",
        "|-----|----|-----|-----|-----|---------|",
    ]
    for mname, blk in (mix.get("mixes") or {}).items():
        lad = blk["ladder"]
        cells = []
        for n in (9, 12, 15, 18):
            s = lad.get(n) or lad.get(str(n))
            cells.append(f"{s['median_inv_p']:.3f}/{s['B90_inv_p']:.3f}" if s else "—")
        lines.append(
            f"| {mname} | " + " | ".join(cells) + f" | **{blk.get('promote')}** |"
        )
    lines += ["", "## #3b CVaR_q80 retrain on L44", ""]
    cvar = results.get("cvar") or {}
    if cvar:
        lad = cvar.get("ladder") or {}
        lines.append(
            f"best step {cvar.get('train_best_step')} score={cvar.get('train_score'):.4f}; "
            f"promote=**{cvar.get('promote')}**"
        )
        lines.append("")
        lines.append("| n | CVaR Med/B90 | champ |")
        lines.append("|---|--------------|-------|")
        for n in (9, 12, 15, 18):
            s = lad.get(n) or lad.get(str(n)) or {}
            lines.append(
                f"| {n} | {s.get('median_inv_p', float('nan')):.3f}/"
                f"{s.get('B90_inv_p', float('nan')):.3f} | "
                f"{CHAMP[n]['Med']:.3f}/{CHAMP[n]['B90']:.3f} |"
            )
    else:
        lines.append("*(skipped)*")

    lines += ["", "## #4 Freedom span_w @ L44", ""]
    fr = results.get("freedom") or {}
    if fr:
        lines.append(
            f"w_span softplus={fr.get('w_span_softplus')}; "
            f"promote=**{fr.get('promote')}** gates={fr.get('gates')}"
        )
        lines.append("")
        lines.append("| n | freedom Med/B90 | champ | thr (0.98×) |")
        lines.append("|---|-----------------|-------|-------------|")
        lad = fr.get("ladder") or {}
        for n in (9, 12, 15, 18):
            s = lad.get(str(n)) or lad.get(n) or {}
            thr = CHAMP[n]["Med"] * FREE_MED_MULT
            lines.append(
                f"| {n} | {s.get('median_inv_p', float('nan')):.3f}/"
                f"{s.get('B90_inv_p', float('nan')):.3f} | "
                f"{CHAMP[n]['Med']:.3f}/{CHAMP[n]['B90']:.3f} | {thr:.3f} |"
            )
    else:
        lines.append("*(skipped)*")

    lines += [
        "",
        "## Verdict",
        "",
        f"- mixture any promote: **{any(b.get('promote') for b in (mix.get('mixes') or {}).values())}**",
        f"- cvar promote: **{cvar.get('promote')}**",
        f"- freedom promote: **{fr.get('promote')}**",
        "",
        "Champ L44 remains sealed unless a promote fired (none expected to overwrite without gate).",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip_mixture", action="store_true")
    ap.add_argument("--skip_cvar", action="store_true")
    ap.add_argument("--skip_freedom", action="store_true")
    ap.add_argument("--cvar_steps", type=int, default=80)
    ap.add_argument("--freedom_steps", type=int, default=60)
    ap.add_argument("--B18", type=int, default=1000, help="B for n18 (mixture/cvar/freedom)")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    ns = [9, 12, 15, 18]
    B_by_n = {9: 1000, 12: 1000, 15: 1000, 18: args.B18}

    results = {
        "champ": CHAMP,
        "gates": {
            "mixture_B90_n18": B90_N18_THR,
            "mixture_Med_mult": MED_MULT,
            "freedom_Med_mult": FREE_MED_MULT,
        },
        "honesty": "Uc=exp(-iγ Hc); no L48+; no blind",
        "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }

    if not args.skip_mixture:
        log("######## #3a MIXTURE ########")
        results["mixture"] = run_mixture(W, args.nf, device, ns, B_by_n)
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
        write_md(results)

    if not args.skip_cvar:
        log("######## #3b CVaR L44 ########")
        best = train_cvar_L44(W, args.nf, device, steps=args.cvar_steps)
        torch.save(best["state"], OUT / "cvar_L44_params.pt")
        lad = eval_schedule(
            W,
            args.nf,
            device,
            best["state"]["gamma"],
            best["state"]["phi"],
            best["state"]["L"],
            ns,
            B_by_n,
        )
        prom, gates = promote_mix_style(lad, ns)
        results["cvar"] = {
            "train_best_step": best["step"],
            "train_score": best["score"],
            "ladder": {str(k): v for k, v in lad.items()},
            "promote": prom,
            "gates": gates,
        }
        if prom:
            # do NOT overwrite sealed champ unless explicitly promoting — save side file
            torch.save(best["state"], OUT / "PROMOTE_cvar_L44_params.pt")
            log("CVaR PROMOTE candidate saved (champ not overwritten)")
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
        write_md(results)

    if not args.skip_freedom:
        log("######## #4 FREEDOM ########")
        results["freedom"] = run_freedom(
            W, args.nf, device, steps=args.freedom_steps, scale_B_by_n=B_by_n
        )
        if results["freedom"]["promote"]:
            log("FREEDOM PROMOTE candidate (champ not overwritten)")
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
        write_md(results)

    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    write_md(results)
    log("DONE", OUT / "RESULTS.md")


if __name__ == "__main__":
    main()

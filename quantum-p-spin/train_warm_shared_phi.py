"""Re-warm shared γ + 4-shell φ from shared-γ+β champs (φ_k=-βk).

Zach: cold shared-γ+φ(5) underperformed shared-γ+β; φ_k=-βk nests, so
warm from champ β then L-BFGS should recover ≥ shared-γ+β (or expose that
shared-across-layers φ cannot nest per-layer β).

Protocol: B_train=64, B_hold=32, Γ=0.5, 20 LBFGS rounds, seeds 0–2.
n6 L∈{1,2,4,8}, n9 L=8.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG.parent))
sys.path.insert(0, str(PKG))

from pspin import make_dataset, hc_torch, gs_torch  # noqa: E402
from circuit_snap import B_SHELL, make_ansatz, fidelity  # noqa: E402
from circuit_fourier import shell_phi_from_betas, init_fourier_angles  # noqa: E402

OUT = PKG / "runs/compare"
CKPT_DIR = OUT / "shared_gb_ckpts"
FOURIER_DIR = OUT / "20261105_G0.5_qaoa_vs_snap/fourier"

Gamma = 0.5
B_train, B_hold, rounds, batch = 64, 32, 20, 32
SEEDS = (0, 1, 2)
JOBS = [(6, L) for L in (1, 2, 4, 8)] + [(9, 8)]

# prior cold / free baselines (for md)
FREE_N9 = 0.43800629675388336
COLD = {
    (9, 8): {"free": FREE_N9, "gb": 0.39216148853302, "gphi_cold": 0.3277926743030548,
             "gb_seeds": [0.39216148853302, 0.3896046578884125, 0.3883666545152664],
             "gphi_cold_seeds": [0.3245591074228287, 0.3277926743030548, 0.32676592469215393]},
    (6, 1): {"free": 0.6868760287761688, "gb": 0.6868760287761688, "gphi_cold": 0.6834830343723297},
    (6, 2): {"free": 0.7411412596702576, "gb": 0.7102271020412445, "gphi_cold": 0.6996},  # filled below
    (6, 4): {"free": 0.8179, "gb": 0.7843, "gphi_cold": 0.7094},
    (6, 8): {"free": 0.9203, "gb": 0.8529, "gphi_cold": 0.7150},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[warm-φ]", *a, flush=True)


def pack(insts, device):
    return [(hc_torch(it, device), gs_torch(it, device), it.E0) for it in insts]


@torch.no_grad()
def medF(sim, gamma_vec, phi, packed):
    Fs = [float(fidelity(sim, Hc, gs, gamma_vec, phi).cpu()) for Hc, gs, _ in packed]
    arr = np.asarray(Fs, dtype=np.float64)
    return float(np.median(arr)), float(arr.mean())


def expand_g(g_scalar, L):
    return g_scalar.expand(L).contiguous()


def expand_phi_shared(phi4, L):
    return phi4.unsqueeze(0).expand(L, -1).contiguous()


def ckpt_path(n, L, seed):
    return CKPT_DIR / f"shared_gb_n{n}_L{L}_G{Gamma}_seed{seed}.pt"


def train_shared_gb(n, L, seed, trains, holds, sim, device):
    """1 shared γ + β[L]. Saves beta."""
    path = ckpt_path(n, L, seed)
    if path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        log(f"reuse gb n{n} L{L} seed{seed} hold={ck['hold_med']:.4f}")
        return ck

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    beta0, gamma0 = init_fourier_angles(L, seed=seed)
    beta = torch.nn.Parameter(beta0.clone().to(device))
    g_scalar = torch.nn.Parameter(
        torch.tensor(float(gamma0.mean()), dtype=torch.float32, device=device)
    )
    best = {
        "hold_med": -1.0,
        "hold_mean": None,
        "train_med": -1.0,
        "round": -1,
        "g_scalar": None,
        "beta": None,
    }
    for r in range(rounds):
        opt = torch.optim.LBFGS(
            [beta, g_scalar],
            lr=0.25,
            max_iter=8,
            history_size=16,
            line_search_fn="strong_wolfe",
        )
        idx = rng.choice(len(trains), size=min(batch, len(trains)), replace=False)

        def closure():
            opt.zero_grad()
            gvec = expand_g(g_scalar, L)
            phi = shell_phi_from_betas(beta)
            losses = [
                1.0 - fidelity(sim, trains[int(j)][0], trains[int(j)][1], gvec, phi)
                for j in idx
            ]
            lt = torch.stack(losses).mean()
            lt.backward()
            torch.nn.utils.clip_grad_norm_([beta, g_scalar], 5.0)
            return lt

        try:
            opt.step(closure)
        except Exception as e:
            log(f"gb n{n} L{L} seed{seed} r{r} skip {e}")
            continue
        with torch.no_grad():
            gvec = expand_g(g_scalar.detach(), L)
            phi = shell_phi_from_betas(beta.detach())
        tr_med, _ = medF(sim, gvec, phi, trains)
        ho_med, ho_mean = medF(sim, gvec, phi, holds)
        if ho_med > best["hold_med"]:
            best = {
                "hold_med": ho_med,
                "hold_mean": ho_mean,
                "train_med": tr_med,
                "round": r,
                "g_scalar": float(g_scalar.detach().cpu()),
                "beta": beta.detach().cpu().clone(),
            }
        if r % 5 == 0 or r + 1 == rounds:
            log(
                f"gb n{n} L{L} seed{seed} r{r} g={float(g_scalar.detach()):+.4f} "
                f"tr={tr_med:.4f} hold={ho_med:.4f} best={best['hold_med']:.4f}"
            )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "n": n,
            "L": L,
            "Gamma": Gamma,
            "seed": seed,
            "hold_med": best["hold_med"],
            "hold_mean": best["hold_mean"],
            "train_med": best["train_med"],
            "best_round": best["round"],
            "g_scalar": best["g_scalar"],
            "beta": best["beta"],
            "n_params": L + 1,
        },
        path,
    )
    log(f"saved {path} hold={best['hold_med']:.4f}")
    return torch.load(path, map_location="cpu", weights_only=False)


def warm_shared_phi(n, L, seed, gb_ck, trains, holds, sim, device):
    """Init φ_k = mean_ℓ(-β_ℓ k), g=g*; free {γ, φ0..3} via LBFGS."""
    beta = gb_ck["beta"].float().to(device)
    g0 = float(gb_ck["g_scalar"])
    phi_layers = shell_phi_from_betas(beta)  # (L,4)
    phi_shared0 = phi_layers.mean(dim=0)  # (4,) projection onto shared-φ

    # control: frozen per-layer linear φ (exact shared-γ+β)
    with torch.no_grad():
        gvec = torch.full((L,), g0, dtype=torch.float32, device=device)
        ctrl_med, ctrl_mean = medF(sim, gvec, phi_layers, holds)
        # also eval shared projection before train
        warm0_med, _ = medF(sim, gvec, expand_phi_shared(phi_shared0, L), holds)

    g_scalar = torch.nn.Parameter(torch.tensor(g0, dtype=torch.float32, device=device))
    phi_param = torch.nn.Parameter(phi_shared0.detach().clone())
    torch.manual_seed(seed + 10_000)
    np.random.seed(seed + 10_000)
    rng = np.random.default_rng(seed + 10_000)

    best = {
        "hold_med": -1.0,
        "hold_mean": None,
        "train_med": -1.0,
        "round": -1,
        "g_scalar": None,
        "phi": None,
    }
    # record init as candidate
    best = {
        "hold_med": warm0_med,
        "hold_mean": None,
        "train_med": warm0_med,
        "round": -1,
        "g_scalar": g0,
        "phi": phi_shared0.detach().cpu().clone(),
    }

    for r in range(rounds):
        opt = torch.optim.LBFGS(
            [g_scalar, phi_param],
            lr=0.25,
            max_iter=8,
            history_size=16,
            line_search_fn="strong_wolfe",
        )
        idx = rng.choice(len(trains), size=min(batch, len(trains)), replace=False)

        def closure():
            opt.zero_grad()
            gvec = expand_g(g_scalar, L)
            phi = expand_phi_shared(phi_param, L)
            losses = [
                1.0 - fidelity(sim, trains[int(j)][0], trains[int(j)][1], gvec, phi)
                for j in idx
            ]
            lt = torch.stack(losses).mean()
            lt.backward()
            torch.nn.utils.clip_grad_norm_([g_scalar, phi_param], 5.0)
            return lt

        try:
            opt.step(closure)
        except Exception as e:
            log(f"warm n{n} L{L} seed{seed} r{r} skip {e}")
            continue
        with torch.no_grad():
            gvec = expand_g(g_scalar.detach(), L)
            phi = expand_phi_shared(phi_param.detach(), L)
        tr_med, _ = medF(sim, gvec, phi, trains)
        ho_med, ho_mean = medF(sim, gvec, phi, holds)
        if ho_med > best["hold_med"]:
            best = {
                "hold_med": ho_med,
                "hold_mean": ho_mean,
                "train_med": tr_med,
                "round": r,
                "g_scalar": float(g_scalar.detach().cpu()),
                "phi": phi_param.detach().cpu().clone(),
            }
        if r % 5 == 0 or r + 1 == rounds:
            log(
                f"warm n{n} L{L} seed{seed} r{r} g={float(g_scalar.detach()):+.4f} "
                f"tr={tr_med:.4f} hold={ho_med:.4f} best={best['hold_med']:.4f}"
            )

    return {
        "hold_med": best["hold_med"],
        "hold_mean": best["hold_mean"],
        "train_med": best["train_med"],
        "best_round": best["round"],
        "g_scalar": best["g_scalar"],
        "phi": best["phi"].tolist() if best["phi"] is not None else None,
        "warm0_hold": warm0_med,
        "control_freeze_linear_hold": ctrl_med,
        "control_freeze_linear_mean": ctrl_mean,
        "gb_hold": float(gb_ck["hold_med"]),
        "n_params": 1 + B_SHELL,
    }


def free_n6_baselines():
    rows = json.loads((FOURIER_DIR / "train_results.json").read_text())
    out = {}
    for L in (1, 2, 4, 8):
        rs = [r for r in rows if r["L"] == L and r["mode"] == "n6" and r["method"] == "fourier"]
        by_seed = {r["seed"]: r["hold"]["6"]["median_F"] for r in rs}
        out[L] = max(by_seed.values()) if by_seed else None
    return out


def load_cold_n6():
    p = OUT / "n6_shared_gamma_sweep.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    cold = {}
    for L_s, v in data.get("summary", {}).items():
        L = int(L_s)
        cold[L] = {
            "free": v["free"]["best"],
            "gb": v["shared_g_beta"]["best"],
            "gphi_cold": v["shared_g_shared_phi"]["best"],
            "gb_by_seed": v["shared_g_beta"]["by_seed"],
            "gphi_cold_by_seed": v["shared_g_shared_phi"]["by_seed"],
        }
    return cold


def main():
    device = torch.device("cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    free6 = free_n6_baselines()
    cold6 = load_cold_n6()
    log(f"free n6: {free6}")
    log("cold n6: " + str({k: round(v["gphi_cold"], 4) for k, v in cold6.items()}))

    data_cache = {}
    rows = []
    summary = {}

    for n, L in JOBS:
        key = (n, Gamma)
        if key not in data_cache:
            log(f"build data n={n}")
            tr = make_dataset(n, Gamma, B_train, seed0=1000 * n + int(round(10 * Gamma)) + 0)
            ho = make_dataset(n, Gamma, B_hold, seed0=1000 * n + int(round(10 * Gamma)) + 50_000)
            data_cache[key] = (pack(tr, device), pack(ho, device))
        trains, holds = data_cache[key]
        sim = make_ansatz(n, L, device)
        log(f"=== n={n} L={L} ===")

        gb_rows, warm_rows = [], []
        for seed in SEEDS:
            gb = train_shared_gb(n, L, seed, trains, holds, sim, device)
            gb_rows.append(
                {
                    "case": "shared_g_beta",
                    "n": n,
                    "L": L,
                    "n_params": L + 1,
                    "seed": seed,
                    "hold_med": float(gb["hold_med"]),
                    "g_scalar": float(gb["g_scalar"]),
                    "beta": gb["beta"].tolist(),
                }
            )
            rows.append(gb_rows[-1])

            w = warm_shared_phi(n, L, seed, gb, trains, holds, sim, device)
            warm_rows.append(
                {
                    "case": "shared_g_shared_phi_warm",
                    "n": n,
                    "L": L,
                    "n_params": 1 + B_SHELL,
                    "seed": seed,
                    **{k: w[k] for k in w},
                    "delta_vs_gb": w["hold_med"] - float(gb["hold_med"]),
                    "delta_vs_control": w["hold_med"] - w["control_freeze_linear_hold"],
                    "control_minus_gb": w["control_freeze_linear_hold"] - float(gb["hold_med"]),
                }
            )
            rows.append(warm_rows[-1])
            log(
                f"DONE warm n{n} L{L} seed{seed} "
                f"gb={gb['hold_med']:.4f} ctrl={w['control_freeze_linear_hold']:.4f} "
                f"warm0={w['warm0_hold']:.4f} warm*={w['hold_med']:.4f} "
                f"Δgb={w['hold_med']-float(gb['hold_med']):+.4f}"
            )
            (OUT / "shared_gamma_shell_snap_warm_partial.json").write_text(
                json.dumps(rows, indent=2)
            )

        gb_best = max(gb_rows, key=lambda r: r["hold_med"])
        warm_best = max(warm_rows, key=lambda r: r["hold_med"])
        free = FREE_N9 if n == 9 else free6[L]
        cold_phi = None
        if n == 9:
            cold_phi = COLD[(9, 8)]["gphi_cold"]
        elif L in cold6:
            cold_phi = cold6[L]["gphi_cold"]

        summary[f"n{n}_L{L}"] = {
            "n": n,
            "L": L,
            "free": free,
            "shared_g_beta": {
                "best": gb_best["hold_med"],
                "n_params": L + 1,
                "by_seed": {r["seed"]: r["hold_med"] for r in gb_rows},
                "g_scalar": gb_best["g_scalar"],
            },
            "shared_g_shared_phi_warm": {
                "best": warm_best["hold_med"],
                "n_params": 1 + B_SHELL,
                "by_seed": {r["seed"]: r["hold_med"] for r in warm_rows},
                "warm0_by_seed": {r["seed"]: r["warm0_hold"] for r in warm_rows},
                "control_by_seed": {
                    r["seed"]: r["control_freeze_linear_hold"] for r in warm_rows
                },
                "g_scalar": warm_best["g_scalar"],
                "phi": warm_best["phi"],
            },
            "gphi_cold": cold_phi,
            "recovers_ge_gb": warm_best["hold_med"] + 1e-6 >= gb_best["hold_med"],
            "delta_warm_minus_gb": warm_best["hold_med"] - gb_best["hold_med"],
            "delta_warm_minus_cold": (
                None if cold_phi is None else warm_best["hold_med"] - cold_phi
            ),
        }

    out = {
        "Gamma": Gamma,
        "B_train": B_train,
        "B_hold": B_hold,
        "rounds": rounds,
        "jobs": [{"n": n, "L": L} for n, L in JOBS],
        "rows": rows,
        "summary": summary,
        "note": (
            "warm init: φ_shared = mean_ℓ shell_phi_from_betas(β_ℓ); g=g* from shared-γ+β. "
            "control_freeze_linear = eval champ with φ_{ℓ,k}=-β_ℓ k (should == gb hold)."
        ),
    }
    (OUT / "shared_gamma_shell_snap_warm.json").write_text(json.dumps(out, indent=2))
    write_md(out, cold6, free6)
    log("done")


def write_md(out, cold6, free6):
    s = out["summary"]
    lines = [
        "# Shared γ + shell SNAP vs QAOA",
        "",
        "Protocol: B_train=64, B_hold=32, cost 1−F, 20 LBFGS rounds, seeds 0–2, Γ=0.5.",
        "",
        "## Warm re-init (Zach)",
        "",
        "Cold shared-γ+shared-φ(5) underperformed shared-γ+β. Nested manifold "
        "`φ_k=-β·k` should let free φ recover ≥ shared-γ+β if optim lands on it.",
        "",
        "Warm protocol:",
        "1. Train/reuse shared-γ+β champs (save β_ℓ, g*).",
        "2. Init shared φ_k = mean_ℓ(−β_ℓ·k), keep g*.",
        "3. L-BFGS free `{γ, φ_0..3}` (5 params).",
        "4. Control: freeze per-layer `φ_{ℓ,k}=-β_ℓ k` (exact shared-γ+β eval).",
        "",
        "### n=9, L=8",
        "",
        "| case | params | best hold MedF | Δ vs free | Δ vs shared-γ+β | seed holds | notes |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    n9 = s["n9_L8"]
    free = n9["free"]
    gb = n9["shared_g_beta"]["best"]
    warm = n9["shared_g_shared_phi_warm"]["best"]
    cold = n9["gphi_cold"]
    # by_seed keys may be int after json roundtrip — handle both
    def seeds_of(d):
        return [d[i] if i in d else d[str(i)] for i in SEEDS]

    gb_seeds = seeds_of(n9["shared_g_beta"]["by_seed"])
    warm_seeds = seeds_of(n9["shared_g_shared_phi_warm"]["by_seed"])
    ctrl_seeds = seeds_of(n9["shared_g_shared_phi_warm"]["control_by_seed"])
    warm0_seeds = seeds_of(n9["shared_g_shared_phi_warm"]["warm0_by_seed"])

    lines += [
        f"| FREE QAOA (γ_ℓ+β_ℓ) | 16 | {free:.4f} | 0.0000 | {free-gb:+.4f} | — | existing fourier L8 n9 |",
        f"| SHARED γ + β_ℓ | 9 | {gb:.4f} | {gb-free:+.4f} | 0.0000 | "
        + ", ".join(f"{x:.4f}" for x in gb_seeds)
        + " | retrained+saved β |",
        f"| SHARED γ + shared φ_k **cold** | 5 | {cold:.4f} | {cold-free:+.4f} | {cold-gb:+.4f} | "
        + "0.3246, 0.3278, 0.3268 | prior cold init |",
        f"| SHARED γ + shared φ_k **warm** | 5 | {warm:.4f} | {warm-free:+.4f} | {warm-gb:+.4f} | "
        + ", ".join(f"{x:.4f}" for x in warm_seeds)
        + f" | from β-proj; recovers≥gb? **{n9['recovers_ge_gb']}** |",
        f"| control freeze φ=-βk (per-ℓ) | 9 | {max(ctrl_seeds):.4f} | {max(ctrl_seeds)-free:+.4f} | "
        f"{max(ctrl_seeds)-gb:+.4f} | "
        + ", ".join(f"{x:.4f}" for x in ctrl_seeds)
        + " | should ≈ shared-γ+β |",
        "",
        f"Warm0 (proj only, no BFGS): "
        + ", ".join(f"{x:.4f}" for x in warm0_seeds)
        + f" → best {max(warm0_seeds):.4f}.",
        "",
        "### Takeaway (n9 L8)",
        "",
        f"- Warm shared-φ hold **{warm:.4f}** vs cold **{cold:.4f}** "
        f"(Δcold {warm-cold:+.4f}) vs shared-γ+β **{gb:.4f}** "
        f"(Δgb {warm-gb:+.4f}).",
        f"- Recovers ≥ shared-γ+β: **{n9['recovers_ge_gb']}**.",
        f"- Control freeze-linear max hold {max(ctrl_seeds):.4f} "
        f"(Δ vs gb {max(ctrl_seeds)-gb:+.6f}) — sanity that φ=-βk reproduces gb.",
        "",
        "Note: shared-across-layers φ (5 params) nests **shared γ + one β** (2 params), "
        "not per-layer β_ℓ. Projection mean_ℓ(−β_ℓ k) can sit below the gb manifold when "
        "β_ℓ vary; free φ may still not reach per-layer-β fidelity.",
        "",
        "## n=6, L ∈ {1,2,4,8}",
        "",
        "| L | free | shared-γ+β | φ cold | φ warm | φ warm0 | ctrl freeze | Δ(warm−gb) | recovers≥gb? |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for L in (1, 2, 4, 8):
        k = f"n6_L{L}"
        v = s[k]
        gb = v["shared_g_beta"]["best"]
        warm = v["shared_g_shared_phi_warm"]["best"]
        cold = v["gphi_cold"] if v["gphi_cold"] is not None else float("nan")
        warm0 = max(seeds_of(v["shared_g_shared_phi_warm"]["warm0_by_seed"]))
        ctrl = max(seeds_of(v["shared_g_shared_phi_warm"]["control_by_seed"]))
        free = v["free"]
        lines.append(
            f"| {L} | {free:.4f} | {gb:.4f} | {cold:.4f} | {warm:.4f} | {warm0:.4f} | "
            f"{ctrl:.4f} | {warm-gb:+.4f} | {v['recovers_ge_gb']} |"
        )

    lines += [
        "",
        "### Takeaway (n6)",
        "",
    ]
    for L in (1, 2, 4, 8):
        v = s[f"n6_L{L}"]
        cold_s = "n/a" if v["gphi_cold"] is None else f"{v['gphi_cold']:.4f}"
        lines.append(
            f"- L={L}: free **{v['free']:.4f}**, gb **{v['shared_g_beta']['best']:.4f}**, "
            f"cold φ **{cold_s}**, warm φ **{v['shared_g_shared_phi_warm']['best']:.4f}** "
            f"(Δgb {v['delta_warm_minus_gb']:+.4f}, recovers≥gb={v['recovers_ge_gb']})."
        )

    lines += [
        "",
        "Artifacts: `shared_gamma_shell_snap_warm.json`, `shared_gb_ckpts/`, "
        "`shared_gamma_shell_snap.json` (cold), `n6_shared_gamma_sweep.json`, "
        "`fixed_gamma_ablation.json`.",
        "",
    ]
    md = OUT / "shared_gamma_shell_snap.md"
    md.write_text("\n".join(lines))
    log(f"wrote {md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

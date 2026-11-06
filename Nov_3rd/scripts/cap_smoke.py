#!/usr/bin/env python3
"""Eval-only smoke for clause-aware cavity compilation (CAP).

Frozen champs, fresh CORE B=1000, 11 permutations each:
  identity, π_max, π_min, and 8 fixed random permutations.

U_c and the Hamming-shell SNAP are not modified. Each permutation is
applied as f'(x) = f(π^{-1} x). Reported Med is 1/p_succ on that
relabeled table, which is success of the original (un-permuted) formula.

QAOA is permutation-invariant. The correctness gate is
|Med_QAOA(π) − Med_QAOA(identity)| < 1e-6 at every CORE n.

PASS: for at least one L in {2, 16}, π_max or π_min has n12 and n15
Med strictly below the range of the 8 random permutations.
KILL: both candidates sit inside that range at n12 and n15 for both L.
No retrain on KILL.
"""
from __future__ import annotations

import argparse
import json
import os
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
from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.cavity_packing import (
    CAP_RANDOM_SEED,
    N_RANDOM,
    cooccurrence,
    identity_perm,
    pack_batch,
    packing_score,
    permute_energy,
    permute_energy_instances,
    random_permutations,
)
from Nov_3rd.data_io import load_split_E, log
from explore_OCT29.lib.qaoa import inv_p_from_psucc, qaoa_psucc

EVAL_NS = (6, 9, 12, 15)
WALSH_PT = (
    S.OUT
    / "beat_qaoa_12h"
    / "train69"
    / "ucx_bit_a"
    / "walsh_degree"
    / "params_L2_seed1_B100.pt"
)
POWER_PT = (
    S.OUT
    / "beat_qaoa_12h"
    / "train69_ladder_fresh250"
    / "power_p"
    / "params_L16_seed0.pt"
)

# Frozen-champ identity Med on this scale1000 (eval json / RESULTS).
# QAOA column is the sealed ladder ref used for Walsh knee R≈0.9567.
PUBLISHED = {
    "walsh_L2": {
        "snap": {6: 5.023059368133545, 9: 11.968106269836426, 12: 28.03518772125244, 15: 72.79664993286133},
        "qaoa": {6: 5.32568408059022, 9: 12.641587441913686, 12: 29.606313521137757, 15: 76.08783622676893},
    },
    "power_L16": {
        "snap": {6: 1.1503223776817322, 9: 1.4183018803596497, 12: 1.838893175125122, 15: 2.579069137573242},
        "qaoa": {6: 1.2330892568001541, 9: 1.5879152397288425, 12: 2.0853667641846796, 15: 2.912675823906689},
    },
}


def _qaoa_invs(betas, gammas, E_np: np.ndarray, n: int, *, chunk: int, dtype: torch.dtype) -> np.ndarray:
    """Per-instance 1/p_succ. dtype float32 matches the sealed ladder; float64 is the invariance gate."""
    invs = []
    E_np = np.asarray(E_np)
    B = E_np.shape[0]
    betas_t = torch.tensor(np.asarray(betas, dtype=np.float64), dtype=dtype)
    gammas_t = torch.tensor(np.asarray(gammas, dtype=np.float64), dtype=dtype)
    # qaoa_psucc forces float32 internally. Use a local complex path when dtype is float64.
    if dtype == torch.float32:
        for s0 in range(0, B, chunk):
            Eb = torch.as_tensor(E_np[s0 : s0 + chunk], dtype=torch.float32)
            ps = qaoa_psucc(betas_t, gammas_t, Eb, n)
            invs.append(inv_p_from_psucc(ps.detach().cpu().numpy()))
        return np.concatenate(invs)
    cdtype = torch.complex128
    m = int(round(9.93 * n))
    for s0 in range(0, B, chunk):
        Eb = torch.as_tensor(E_np[s0 : s0 + chunk], dtype=torch.float64)
        Bsz, dim = Eb.shape
        Cn = Eb * (float(n) / float(m))
        psi = torch.full((Bsz, dim), 1.0 / np.sqrt(dim), dtype=cdtype)
        for ell in range(int(gammas_t.numel())):
            psi = psi * torch.exp(-1j * gammas_t[ell].to(cdtype) * Cn.to(cdtype))
            beta = betas_t[ell]
            c = torch.cos(beta / 2).to(cdtype)
            s = (-1j * torch.sin(beta / 2)).to(cdtype)
            for q in range(n):
                v = psi.reshape(Bsz, 1 << q, 2, 1 << (n - 1 - q))
                a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
                psi = torch.stack([c * a0 + s * a1, s * a0 + c * a1], dim=2).reshape(Bsz, -1)
        probs = psi.real.square() + psi.imag.square()
        sat = (Eb == 0).to(dtype=probs.dtype)
        ps = (probs * sat).sum(dim=-1).detach().cpu().numpy()
        invs.append(inv_p_from_psucc(ps))
    return np.concatenate(invs)


def _summarize(invs: np.ndarray) -> dict:
    a = np.asarray(invs, dtype=np.float64)
    return {
        "Med": float(np.median(a)),
        "B90": float(np.quantile(a, 0.90)),
        "B": int(a.size),
    }


def load_core(n: int, B: int):
    E_t, key = load_split_E(n, "scale1000", B, torch.device("cpu"), data_dir=S.SCALE1000_DIR)
    path = S.SCALE1000_DIR / f"data_n{n}.npz"
    z = np.load(path)
    cls = np.asarray(z["scale1000_cls_v"][:B])
    E = np.asarray(z["scale1000_E"][:B])
    if int(E_t.shape[0]) != int(E.shape[0]):
        raise RuntimeError(f"n{n}: loader B={len(E_t)} npz B={len(E)} key={key}")
    if not np.array_equal(E_t.detach().cpu().numpy().astype(np.int16), E.astype(np.int16)):
        raise RuntimeError(f"n{n}: loader E != scale1000_E (key={key})")
    return E, E_t, cls


def strategies_for_n(cls_v: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """name → π with shape (n,) or (B, n) for the adaptive rules."""
    w = cooccurrence(cls_v, n)
    t0 = time.time()
    pi_max = pack_batch(cls_v, n, True)
    pi_min = pack_batch(cls_v, n, False)
    log(f"n{n}: packed {len(cls_v)} instances in {time.time() - t0:.2f}s")
    out: dict[str, np.ndarray] = {
        "identity": identity_perm(n),
        "pi_max": pi_max,
        "pi_min": pi_min,
    }
    for k, pi in enumerate(random_permutations(n, N_RANDOM, CAP_RANDOM_SEED)):
        out[f"random_{k}"] = pi
    # score sanity — adaptive objective must separate from identity
    def mean_score(pi):
        if pi.ndim == 1:
            return float(np.mean([packing_score(w[b], pi) for b in range(len(w))]))
        return float(np.mean([packing_score(w[b], pi[b]) for b in range(len(w))]))

    scores = {name: mean_score(pi) for name, pi in out.items()}
    if not (scores["pi_max"] > scores["identity"] > scores["pi_min"]):
        raise RuntimeError(f"n{n}: packing scores did not separate {scores}")
    rnd = [scores[f"random_{k}"] for k in range(N_RANDOM)]
    if scores["pi_max"] <= max(rnd) or scores["pi_min"] >= min(rnd):
        raise RuntimeError(f"n{n}: objective not outside random range {scores}")
    return out, scores, w


def apply_strategy(E: np.ndarray, pi: np.ndarray) -> np.ndarray:
    if pi.ndim == 1:
        return permute_energy(E, pi)
    return permute_energy_instances(E, pi)


def champ_angles(which: str, n: int, device):
    if which == "walsh_L2":
        st = torch.load(WALSH_PT, map_location="cpu", weights_only=False)
        gamma = st["gamma_w"].to(device)
        phi = phi_from_c(st["c1"].to(device), st["c2"].to(device), st["c3"].to(device))
        betas = st["betas_qaoa"]
        gammas = st["gammas_qaoa"]
        L = 2
    elif which == "power_L16":
        st = torch.load(POWER_PT, map_location="cpu", weights_only=False)
        p = float(st["p"])
        a = (8.0 / float(n - 1)) ** p
        gamma = st["gamma"].to(device)
        phi = phi_from_c(st["c1"].to(device), st["c2"].to(device) * a, st["c3"].to(device) * a)
        betas = st["betas_qaoa"]
        gammas = st["gammas_qaoa"]
        L = 16
    else:
        raise KeyError(which)
    return L, gamma, phi, betas, gammas


@torch.no_grad()
def eval_snap(n: int, L: int, gamma, phi, E_perm: np.ndarray, *, every: int = 200) -> dict:
    sim = make_sim(n, L, torch.device("cpu"))
    invs = np.empty(len(E_perm), dtype=np.float64)
    leaks = np.empty(len(E_perm), dtype=np.float64)
    t0 = time.time()
    for i in range(len(E_perm)):
        Et = torch.from_numpy(np.ascontiguousarray(E_perm[i])).to(dtype=torch.float32)
        out = sim.forward_nov3rd(Et, gamma, phi)
        invs[i] = float(out["inv_p"])
        leaks[i] = float(out["p_leak"])
        if every and (i + 1) % every == 0:
            log(f"  snap n{n} L{L} {i+1}/{len(E_perm)} Med≈{np.median(invs[:i+1]):.4f}")
    sm = _summarize(invs)
    sm["mean_leak"] = float(leaks.mean())
    sm["wall_s"] = time.time() - t0
    return sm


def _range_better(med: float, rnd: list[float]) -> dict:
    lo = float(min(rnd))
    hi = float(max(rnd))
    # Strictly outside, lower is better. 1e-9 absorbs median tie noise, not a real effect.
    better = bool(med < lo - 1e-9)
    inside = bool((lo - 1e-9) <= med <= (hi + 1e-9))
    return {"Med": med, "random_min": lo, "random_max": hi, "better_outside": better, "inside": inside}


def judge(snap: dict) -> dict:
    """snap[which][perm][str(n)] = summary."""
    per = []
    any_pass = False
    all_inside = True
    for which in ("walsh_L2", "power_L16"):
        block = snap[which]
        for cand in ("pi_max", "pi_min"):
            detail = {}
            ok = True
            for n in (12, 15):
                rnd = [block[f"random_{k}"][str(n)]["Med"] for k in range(N_RANDOM)]
                rec = _range_better(block[cand][str(n)]["Med"], rnd)
                detail[str(n)] = rec
                ok = ok and rec["better_outside"]
                if not rec["inside"]:
                    all_inside = False
            if ok:
                any_pass = True
            per.append({"champ": which, "perm": cand, "pass_n12_and_n15": ok, "detail": detail})
    if any_pass:
        verdict = "PASS"
    elif all_inside:
        verdict = "KILL"
    else:
        verdict = "NO_PASS"
    return {"verdict": verdict, "checks": per}


def write_results(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    v = payload["verdict"]["verdict"]
    lines = [
        "# CAP — clause-aware cavity compilation",
        "",
        "Label: **clause-aware cavity compilation**. This is a variable relabeling.",
        "U_c^X and Hamming-shell SNAP are the frozen champs; the circuit sees",
        "f'(x) = f(π^{-1} x). Med is 1/p_succ with success on the original formula",
        "(E'(x) = 0 iff un-permuted E is satisfied). Zero new parameters.",
        "Identity π is exactly the published champ.",
        "",
        f"**Verdict: {v}**",
        "",
        "PASS requires π_max or π_min Med strictly below the 8-random range",
        "at both n=12 and n=15, for at least one of L=2 (Walsh seed1) and",
        "L=16 (fresh250 power_p). KILL means that effect stays inside the",
        "random range at n=12 and n=15 for both L. No retrain on KILL.",
        "",
        "## Packing objective (mean Σ pair-weights inside cavities)",
        "",
        "π_max / π_min are instance-adaptive. Identity and the 8 random",
        f"permutations are one π per n (PCG64 seed {CAP_RANDOM_SEED}+n), shared",
        "across the B instances. The objective separating is a check that the",
        "search ran; it is not the Med gate.",
        "",
    ]
    scores = payload.get("scores", {})
    header = "| n | identity | π_max | π_min | random min | random max |"
    lines += [header, "|---|----------|-------|-------|------------|------------|"]
    for n in EVAL_NS:
        sc = scores[str(n)]
        rnd = [sc[f"random_{k}"] for k in range(N_RANDOM)]
        lines.append(
            f"| {n} | {sc['identity']:.2f} | {sc['pi_max']:.2f} | {sc['pi_min']:.2f} | {min(rnd):.2f} | {max(rnd):.2f} |"
        )
    lines += [
        "",
        "## QAOA invariance",
        "",
        "Gate: |ΔMed| vs identity < 1e-6, float64, same angles as the champ.",
        "Every non-identity permutation passed. The largest absolute gap is below.",
        "",
        "| champ | max |ΔMed| over 10 perms × CORE |",
        "|-------|--------------------------------|",
    ]
    qaoa = payload["qaoa"]
    for which in ("walsh_L2", "power_L16"):
        ident = qaoa[which]["identity"]
        worst = 0.0
        for name, block in qaoa[which].items():
            if name == "identity":
                continue
            for n in EVAL_NS:
                worst = max(worst, abs(block[str(n)]["Med"] - ident[str(n)]["Med"]))
        lines.append(f"| {which} | {worst:.3e} |")
    passed = [c for c in payload["verdict"]["checks"] if c["pass_n12_and_n15"]]
    lines += ["", "Smoke checks that cleared n12 and n15 together:", ""]
    if not passed:
        lines.append("- (none)")
    for c in passed:
        lines.append(f"- {c['champ']} / {c['perm']}")
    lines += ["", "## SNAP Med (lower is better)", ""]
    for which, title in (
        ("walsh_L2", "L=2 Walsh-degree seed1 (frozen)"),
        ("power_L16", "L=16 fresh250 power_p (frozen)"),
    ):
        lines += [
            f"### {title}",
            "",
            "| perm | n6 | n9 | n12 | n15 |",
            "|------|----|----|-----|-----|",
        ]
        block = payload["snap"][which]
        order = ["identity", "pi_max", "pi_min"] + [f"random_{k}" for k in range(N_RANDOM)]
        for name in order:
            cells = [f"{block[name][str(n)]['Med']:.4f}" for n in EVAL_NS]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("| perm | n12 vs random | n15 vs random |")
        lines.append("|------|---------------|---------------|")
        for cand in ("pi_max", "pi_min", "identity"):
            bits = []
            for n in (12, 15):
                rnd = [block[f"random_{k}"][str(n)]["Med"] for k in range(N_RANDOM)]
                rec = _range_better(block[cand][str(n)]["Med"], rnd)
                flag = "BETTER" if rec["better_outside"] else ("inside" if rec["inside"] else "worse-outside")
                bits.append(f"{rec['Med']:.4f} [{rec['random_min']:.4f}, {rec['random_max']:.4f}] {flag}")
            lines.append(f"| {cand} | {bits[0]} | {bits[1]} |")
        lines.append("")
    lines += [
        "## Identity reproduction",
        "",
        "Identity Med must match the published frozen-champ CORE eval.",
        "",
    ]
    f32 = payload.get("qaoa_f32_identity", {})
    for which in ("walsh_L2", "power_L16"):
        for n in EVAL_NS:
            got = payload["snap"][which]["identity"][str(n)]["Med"]
            pub = PUBLISHED[which]["snap"][n]
            lines.append(f"- {which} SNAP n{n}: got {got:.6f} published {pub:.6f} Δ={got - pub:.3e}")
        for n in EVAL_NS:
            block = f32.get(which, {}).get("identity", {}).get(str(n))
            got = float(block["Med"]) if block else float("nan")
            pub = PUBLISHED[which]["qaoa"][n]
            lines.append(
                f"- {which} QAOA float32 n{n}: got {got:.6f} published {pub:.6f} Δ={got - pub:.3e}"
            )
    retrain_md = path.parent / "retrain" / "RETRAIN.md"
    if retrain_md.exists():
        lines += ["", retrain_md.read_text().rstrip(), ""]
    lines += ["", f"Finished: {payload.get('finished', '')}", ""]
    md = path.with_suffix(".md")
    # caller passes results.json; also write RESULTS.md beside it
    (path.parent / "RESULTS.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {path} and RESULTS.md verdict={v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=str, default=str(S.OUT / "beat_qaoa_12h" / "cap"))
    ap.add_argument("--only", type=str, default="", help="walsh_L2 or power_L16 (default both)")
    ap.add_argument("--skip-snap", action="store_true")
    ap.add_argument("--skip-qaoa", action="store_true")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoint.json"
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
    else:
        ckpt = {"scores": {}, "qaoa": {}, "snap": {}, "B": args.B}

    if int(ckpt.get("B", args.B)) != int(args.B):
        raise SystemExit(f"checkpoint B={ckpt.get('B')} != {args.B}; use a fresh --out")

    which_list = [args.only] if args.only else ["walsh_L2", "power_L16"]
    loaded = {}
    strat = {}
    for n in EVAL_NS:
        E, E_t, cls = load_core(n, args.B)
        loaded[n] = (E, E_t, cls)
        st, scores, _w = strategies_for_n(cls, n)
        strat[n] = st
        ckpt["scores"][str(n)] = scores
        log(f"n{n} scores identity={scores['identity']:.2f} max={scores['pi_max']:.2f} min={scores['pi_min']:.2f}")
        ckpt_path.write_text(json.dumps(ckpt, indent=2) + "\n")

    def need(section, which, name, n):
        return str(n) not in ckpt.get(section, {}).get(which, {}).get(name, {})

    def put(section, which, name, n, summary):
        ckpt.setdefault(section, {}).setdefault(which, {}).setdefault(name, {})[str(n)] = summary
        ckpt_path.write_text(json.dumps(ckpt, indent=2) + "\n")

    # QAOA gate in float64. Also record float32 identity against the sealed ref.
    if not args.skip_qaoa:
        for which in which_list:
            ckpt.setdefault("qaoa", {}).setdefault(which, {})
            ckpt.setdefault("qaoa_f32_identity", {}).setdefault(which, {})
            for n in EVAL_NS:
                _L, _g, _p, betas, gammas = champ_angles(which, n, torch.device("cpu"))
                E, _Et, _cls = loaded[n]
                chunk = 64 if n <= 9 else (16 if n <= 12 else 4)
                if need("qaoa_f32_identity", which, "identity", n):
                    t0 = time.time()
                    inv = _qaoa_invs(betas, gammas, E, n, chunk=chunk, dtype=torch.float32)
                    sm = _summarize(inv)
                    sm["wall_s"] = time.time() - t0
                    pub = PUBLISHED[which]["qaoa"][n]
                    sm["published"] = pub
                    sm["abs_delta"] = abs(sm["Med"] - pub)
                    put("qaoa_f32_identity", which, "identity", n, sm)
                    log(f"QAOA f32 {which} n{n} Med={sm['Med']:.6f} pub={pub:.6f} Δ={sm['abs_delta']:.3e}")
                    if sm["abs_delta"] > 1e-4:
                        raise SystemExit(f"identity QAOA drifted from sealed ref at {which} n{n}")
                names = ["identity", "pi_max", "pi_min"] + [f"random_{k}" for k in range(N_RANDOM)]
                for name in names:
                    if not need("qaoa", which, name, n):
                        continue
                    Ep = apply_strategy(E, strat[n][name])
                    t0 = time.time()
                    inv = _qaoa_invs(betas, gammas, Ep, n, chunk=chunk, dtype=torch.float64)
                    sm = _summarize(inv)
                    sm["wall_s"] = time.time() - t0
                    put("qaoa", which, name, n, sm)
                    log(f"QAOA f64 {which} {name} n{n} Med={sm['Med']:.8f} ({sm['wall_s']:.1f}s)")
                ident = ckpt["qaoa"][which]["identity"][str(n)]["Med"]
                for name in names:
                    if name == "identity":
                        continue
                    d = abs(ckpt["qaoa"][which][name][str(n)]["Med"] - ident)
                    if d >= 1e-6:
                        raise SystemExit(
                            f"QAOA invariance failed {which} {name} n{n}: |ΔMed|={d:.3e}"
                        )
                log(f"QAOA gate PASS {which} n{n}")

    if not args.skip_snap:
        for which in which_list:
            ckpt.setdefault("snap", {}).setdefault(which, {})
            names = ["identity", "pi_max", "pi_min"] + [f"random_{k}" for k in range(N_RANDOM)]
            for n in EVAL_NS:
                L, gamma, phi, _b, _g = champ_angles(which, n, torch.device("cpu"))
                E, E_t, _cls = loaded[n]
                for name in names:
                    if not need("snap", which, name, n):
                        log(f"snap skip {which} {name} n{n}")
                        continue
                    if name == "identity":
                        # exact champ path: the loader tensor, not a permuted copy
                        E_use = E_t.detach().cpu().numpy()
                    else:
                        E_use = apply_strategy(E, strat[n][name])
                    log(f"snap {which} {name} n{n} B={len(E_use)}")
                    sm = eval_snap(n, L, gamma, phi, E_use, every=250 if n >= 12 else 0)
                    put("snap", which, name, n, sm)
                    log(
                        f"snap {which} {name} n{n} Med={sm['Med']:.4f} "
                        f"B90={sm['B90']:.4f} leak={sm['mean_leak']:.2e} ({sm['wall_s']:.1f}s)"
                    )
                pub = PUBLISHED[which]["snap"][n]
                got = ckpt["snap"][which]["identity"][str(n)]["Med"]
                if abs(got - pub) > 1e-3:
                    raise SystemExit(f"identity SNAP drifted {which} n{n}: got {got} pub {pub}")

    # Judgement needs both champs. If --only, don't overwrite a partial verdict.
    ready = all(
        str(n) in ckpt.get("snap", {}).get(which, {}).get("pi_max", {})
        and str(n) in ckpt.get("snap", {}).get(which, {}).get("random_7", {})
        for which in ("walsh_L2", "power_L16")
        for n in EVAL_NS
    )
    q_ready = all(
        str(n) in ckpt.get("qaoa", {}).get(which, {}).get("random_7", {})
        for which in ("walsh_L2", "power_L16")
        for n in EVAL_NS
    )
    if ready and q_ready:
        verdict = judge(ckpt["snap"])
        payload = {
            "label": "clause-aware cavity compilation",
            "honesty": "success checked on un-permuted E via f'(x)=f(pi^{-1} x); U_c and SNAP unchanged",
            "B": args.B,
            "eval_ns": list(EVAL_NS),
            "random_seed": CAP_RANDOM_SEED,
            "n_random": N_RANDOM,
            "champs": {
                "walsh_L2": str(WALSH_PT),
                "power_L16": str(POWER_PT),
            },
            "scores": ckpt["scores"],
            "qaoa": ckpt["qaoa"],
            "qaoa_f32_identity": ckpt.get("qaoa_f32_identity", {}),
            "snap": ckpt["snap"],
            "verdict": verdict,
            "retrain": False,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        }
        write_results(out / "results.json", payload)
        log(f"VERDICT {verdict['verdict']}")
    else:
        log("partial checkpoint written; verdict waits for both champs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

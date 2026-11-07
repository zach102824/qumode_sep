#!/usr/bin/env python3
"""Build FC-WS RESULTS.md from eval jsons. Applies Opus Idea 1 kill criteria."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Nov_3rd.fc_ws import CLASS_NAME
from train69_fc_ws import OUT_ROOT, bootstrap_med_diff

EVAL_NS = [6, 9, 12, 15]
KAPPA_ZERO = 0.02


def load_evals(root: Path, tag: str) -> list[dict]:
    rows = []
    for p in sorted(root.rglob(f"eval_{tag}.json")):
        d = json.loads(p.read_text())
        d["_path"] = str(p)
        rows.append(d)
    return rows


def key_of(d: dict) -> tuple:
    return (d.get("warm"), d.get("arm"), int(d.get("seed", -1)), bool(d.get("pool_111", False)), int(d.get("L", -1)))


def med(d, n):
    return float(d["per_n"][str(n)]["Med"])


def R(d, n):
    return float(d["per_n"][str(n)]["R"])


def inv(d, n):
    return d["per_n"][str(n)].get("inv")


def fmt_row(d) -> str:
    cells = []
    for n in EVAL_NS:
        r = d["per_n"][str(n)]
        cells.append(f"{r['Med']:.3f} ({r['R']:.4f})")
    knee = max(R(d, n) for n in EVAL_NS)
    raw = all(r["verdict"] == "BEAT" for r in (d["per_n"][str(n)] for n in EVAL_NS))
    legacy = all(R(d, n) <= 0.98 for n in EVAL_NS)
    return "| " + " | ".join(cells) + f" | {knee:.4f} | {raw} | {legacy} |"


def train_meta(root: Path, L, warm, arm, seed, pool=False) -> dict | None:
    p = root / f"L{L}" / f"warm_{warm}" / f"arm_{arm}" / f"pool{int(pool)}" / f"seed{seed}" / "train.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    root = OUT_ROOT
    lines = []
    lines.append("# FC-WS RESULTS — footprint-class Walsh × commutator-matched shell")
    lines.append("")
    lines.append("Opus Idea 1. Graded H_c (class boosts `(1+δ_C)` on Walsh monomials, labeled as such).")
    lines.append("Train `{6,9}` only (`max_train=100`, w6=1, w9=2). Schedule frozen. Globals: 6 δ + κ (pool off).")
    lines.append("CORE scale1000 regenerated with `sat_gen` SCB=54e9; L=2 QAOA Med matches `qaoa_angles/L2/results.json` exactly.")
    lines.append("Raw BEAT = Med < QAOA Med. Legacy BEAT = R≤0.98 (older gate).")
    lines.append("")
    lines.append("Warm starts:")
    lines.append("- **champ** — L=2 pareto a0.80 s1 (scalar γ + shell). L=16 sealed power_p seed0, learn only the globals.")
    lines.append("- **walsh** — L=2 Walsh-degree γ_w seed1. δ=0, κ=0 recovers that circuit.")
    lines.append("")

    verdict_bits = []

    for tag, title in [("B100", "B=100 smoke (gate slice)"), ("B1000", "B=1000 CORE")]:
        ev = load_evals(root, tag)
        if not ev:
            lines.append(f"## {title}")
            lines.append("")
            lines.append("_no evals_")
            lines.append("")
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| warm | arm | seed | n6 Med (R) | n9 Med (R) | n12 Med (R) | n15 Med (R) | knee R | raw ALL | legacy ALL |")
        lines.append("|------|-----|------|------------|------------|-------------|-------------|--------|---------|------------|")
        for d in ev:
            if int(d.get("L", 2)) != 2 and tag == "B100":
                continue
            w = d.get("warm") or "-"
            lines.append(f"| {w} | {d.get('arm')} | {d.get('seed')} | {fmt_row(d)[2:]}")
        lines.append("")

    # Kill criteria on B=1000 L=2, both warms.
    lines.append("## L=2 kill criteria (B=1000)")
    lines.append("")
    lines.append("κ→0 means |κ|<0.02. Gain is ΔR = R_ref − R_arm (positive = better than the reference).")
    lines.append("Reference for champ-warm is the champ. Reference for walsh-warm is Walsh seed1 (the warm start).")
    lines.append("Bootstrap is paired Med_b − Med_c on the same 1000 instances; c beats b at 95% if the CI lower bound is >0.")
    lines.append("")

    b1000 = {key_of(d): d for d in load_evals(root, "B1000") if int(d.get("L", -1)) == 2}
    # baselines stored warm differently
    def find(warm, arm, seed, L=2):
        # baseline walsh has warm walsh from our writer? check
        for d in load_evals(root, "B1000"):
            if int(d.get("L", -1)) != L:
                continue
            if d.get("arm") == arm and int(d.get("seed", -1)) == seed:
                if arm == "walsh":
                    return d
                if arm == "champ" and warm == "champ":
                    return d
                if d.get("warm") == warm:
                    return d
        return None

    for warm in ("champ", "walsh"):
        lines.append(f"### warm = {warm}")
        lines.append("")
        ref = find("champ", "champ", 0) if warm == "champ" else find("walsh", "walsh", 1)
        if ref is None:
            lines.append("_missing reference_")
            continue
        kappas = []
        seed_rows = []
        n_sig_both = 0
        n_kappa0 = 0
        for seed in (0, 1, 2):
            c = find(warm, "joint", seed)
            b = find(warm, "footprint", seed)
            meta = train_meta(root, 2, warm, "joint", seed, pool=False)
            kap = None if meta is None else float(meta["kappa"])
            kappas.append(kap)
            if kap is not None and abs(kap) < KAPPA_ZERO:
                n_kappa0 += 1
            if c is None or b is None:
                seed_rows.append(f"- seed {seed}: missing eval")
                continue
            boots = {}
            for n in (12, 15):
                ib, ic = inv(b, n), inv(c, n)
                if ib and ic:
                    boots[n] = bootstrap_med_diff(ib, ic)
                else:
                    boots[n] = None
            sig = all(boots[n] and boots[n]["c_better_95"] for n in (12, 15))
            if sig:
                n_sig_both += 1
            # gains vs ref
            def dR(arm, n):
                return R(ref, n) - R(arm, n)
            g9, g15 = dR(c, 9), dR(c, 15)
            fit_train = g15 < g9
            walsh = find("walsh", "walsh", 1)
            beat_b = all(med(c, n) < med(b, n) for n in (12, 15))
            beat_d = walsh is not None and all(med(c, n) < med(walsh, n) for n in (12, 15))
            seed_rows.append(
                f"- seed {seed}: κ={kap:+.4f} δ={None if meta is None else [round(x,3) for x in meta['delta']]} "
                f"hold={None if meta is None else round(meta['holdout_best'],4)} @{None if meta is None else meta['holdout_best_step']}"
            )
            seed_rows.append(
                f"  - Med n12 c/b/d = {med(c,12):.3f} / {med(b,12):.3f} / {med(walsh,12):.3f}; "
                f"n15 c/b/d = {med(c,15):.3f} / {med(b,15):.3f} / {med(walsh,15):.3f}"
            )
            seed_rows.append(
                f"  - point (c)<(b) on n12&n15: {beat_b}; (c)<Walsh-s1 on n12&n15: {beat_d}"
            )
            for n in (12, 15):
                bt = boots[n]
                if bt:
                    seed_rows.append(
                        f"  - boot n{n}: Med_b−Med_c mean {bt['mean_diff']:.3f} CI [{bt['lo']:.3f}, {bt['hi']:.3f}] c_better_95={bt['c_better_95']}"
                    )
            seed_rows.append(
                f"  - ΔR vs ref n9={g9:+.4f} n15={g15:+.4f} n15_gain_smaller_than_n9={fit_train}"
            )
        lines.extend(seed_rows)
        lines.append("")
        kill_kappa = n_kappa0 >= 2
        lines.append(f"Seeds with |κ|<{KAPPA_ZERO}: {n_kappa0}/3. Kill-κ (≥2/3): **{kill_kappa}**.")
        lines.append(f"Seeds where (c) beats (b) at 95% CI on both n12 and n15: {n_sig_both}/3.")
        lines.append("")
        verdict_bits.append((warm, n_kappa0, n_sig_both))

    # Gate on B=100 point estimate, as specified.
    lines.append("## L=2 B=100 gate")
    lines.append("")
    lines.append("Gate: joint beats footprint (same seed) and Walsh-degree seed1 on **both** n12 and n15 Med.")
    lines.append("")
    b100 = load_evals(root, "B100")

    def find100(warm, arm, seed):
        for d in b100:
            if int(d.get("L", -1)) != 2:
                continue
            if d.get("arm") != arm or int(d.get("seed", -1)) != seed:
                continue
            if arm in ("walsh", "champ"):
                return d
            if d.get("warm") == warm:
                return d
        return None

    gate_pass = False
    walsh100 = find100("walsh", "walsh", 1)
    for warm in ("champ", "walsh"):
        for seed in (0, 1, 2):
            c, b = find100(warm, "joint", seed), find100(warm, "footprint", seed)
            if not (c and b and walsh100):
                continue
            ok_b = all(med(c, n) < med(b, n) for n in (12, 15))
            ok_d = all(med(c, n) < med(walsh100, n) for n in (12, 15))
            flag = ok_b and ok_d
            gate_pass = gate_pass or flag
            lines.append(
                f"- warm {warm} seed {seed}: (c)<(b) {ok_b}, (c)<(d seed1) {ok_d} → gate {flag}"
            )
    lines.append("")
    lines.append(f"**Any seed passes the B=100 gate:** {gate_pass}")
    lines.append("")
    lines.append("B=1000 was still run for every L=2 arm. The earlier Walsh study showed this B=100 n15 slice")
    lines.append("sits near R≈1.00 even when the sealed B=1000 knee is ~0.96, so a kill on B=100 n15 alone is not decisive.")
    lines.append("")

    text_path = root / "RESULTS.md"
    # L=16 section filled if present
    ev16 = [d for d in load_evals(root, "B1000") if int(d.get("L", -1)) == 16]
    lines.append("## L=16 frozen power_p schedule")
    lines.append("")
    if not ev16:
        lines.append("_pending_")
    else:
        lines.append("| arm | seed | κ | n6 Med (R) | n9 | n12 | n15 | knee R | raw ALL |")
        lines.append("|-----|------|---|------------|----|-----|-----|--------|---------|")
        for d in ev16:
            meta = train_meta(root, 16, d.get("warm") or "champ", d.get("arm"), int(d.get("seed", 0)))
            kap = "" if meta is None else f"{meta['kappa']:+.3f}"
            cells = " | ".join(f"{d['per_n'][str(n)]['Med']:.3f} ({d['per_n'][str(n)]['R']:.4f})" for n in EVAL_NS)
            knee = max(float(d["per_n"][str(n)]["R"]) for n in EVAL_NS)
            raw = all(d["per_n"][str(n)]["verdict"] == "BEAT" for n in EVAL_NS)
            lines.append(f"| {d.get('arm')} | {d.get('seed')} | {kap} | {cells} | {knee:.4f} | {raw} |")
        lines.append("")
    lines.append("")
    lines.append(f"Class order: {', '.join(CLASS_NAME)}.")
    lines.append("")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {text_path}")


if __name__ == "__main__":
    main()

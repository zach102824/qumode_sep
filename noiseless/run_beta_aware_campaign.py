#!/usr/bin/env python3
"""Orchestrate β-aware fleets A0 → A1 grid → A2 (+ optional extras).

Writes JSON under noiseless/results/ and refreshes BETA_AWARE_SUMMARY.md.
Designed to run unattended for ~12h.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PY = os.environ.get("QUMODE_PYTHON", sys.executable)
_RESULTS = _REPO / "noiseless" / "results"
_SUMMARY = _REPO / "noiseless" / "results" / "BETA_AWARE_SUMMARY.md"
_LOG = _REPO / "logs" / "beta_aware_campaign.log"

BASELINE_MEAN_ABS_BETA = 1.84
BASELINE_SUCCESS = 0.94
SEED = 20260917
WORKERS = max(1, min(7, (os.cpu_count() or 1)))
STEPS = 200
TRIALS = 25
LAYERS = "4"
U = "ck_pi4"
HAM = str(_REPO / "Hamiltonians" / "four_sat")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    _LOG.parent.mkdir(parents=True, exist_ok=True)
    with _LOG.open("a") as f:
        f.write(line + "\n")


def run_fleet(tag: str, *, lambda1: float = 0.0, lambda3: float = 0.0, beta_max: float | None = None,
              layers: str = LAYERS, steps: int = STEPS) -> Path | None:
    cmd = [
        _PY, "-m", "noiseless.run_u_sweep",
        "--ham-dir", HAM,
        "--u-names", U,
        "--layers", layers,
        "--trials", str(TRIALS),
        "--steps", str(steps),
        "--workers", str(WORKERS),
        "--seed", str(SEED),
        "--outdir", str(_RESULTS),
        "--tag", tag,
        "--lambda1", str(lambda1),
        "--lambda3", str(lambda3),
    ]
    if beta_max is not None:
        cmd.extend(["--beta-max", str(beta_max)])
    log(f"START fleet tag={tag} λ1={lambda1} λ3={lambda3} β_max={beta_max} layers={layers} steps={steps}")
    t0 = time.perf_counter()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    proc = subprocess.run(cmd, cwd=str(_REPO), env=env)
    dt = time.perf_counter() - t0
    log(f"END fleet tag={tag} exit={proc.returncode} wall_h={dt/3600:.2f}")
    if proc.returncode != 0:
        return None
    # newest matching summary
    cands = sorted(_RESULTS.glob(f"{tag}_*_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def load_metrics(summary_path: Path) -> dict:
    data = json.loads(summary_path.read_text())
    # Prefer full json for per-trial β if summary lacks detail
    full = Path(str(summary_path).replace("_summary.json", ".json"))
    ranking = data.get("ranking") or []
    row = ranking[0] if ranking else {}
    success = float(row.get("success_rate", 0.0))
    mean_p = float(row.get("mean_p_gs", 0.0))
    mean_b = float(row.get("mean_abs_beta", float("nan")))
    med_b = float(row.get("median_abs_beta", float("nan")))
    mean_max_b = float(row.get("mean_max_abs_beta", float("nan")))
    frac = float("nan")
    if full.exists():
        payload = json.loads(full.read_text())
        recs = [r for r in payload.get("records", []) if r.get("ok")]
        if recs:
            import statistics as stats
            mabs = [float(r.get("mean_abs_beta", float("nan"))) for r in recs]
            maxs = [float(r.get("max_abs_beta", float("nan"))) for r in recs]
            fracs = [float(r.get("frac_over_beta_max", 0.0)) for r in recs]
            mean_b = float(stats.fmean(mabs))
            med_b = float(stats.median(mabs))
            mean_max_b = float(stats.fmean(maxs))
            frac = float(stats.fmean(fracs))
            success = sum(1 for r in recs if r.get("success")) / len(recs)
            mean_p = float(stats.fmean(float(r["p_gs"]) for r in recs))
    args = data.get("args") or {}
    cut = (BASELINE_MEAN_ABS_BETA - mean_b) / BASELINE_MEAN_ABS_BETA if mean_b == mean_b else float("nan")
    return {
        "tag": data.get("tag", summary_path.stem),
        "path": str(summary_path.name),
        "lambda1": args.get("lambda1", 0.0),
        "lambda3": args.get("lambda3", 0.0),
        "beta_max": args.get("beta_max"),
        "layers": args.get("layers"),
        "steps": args.get("steps"),
        "success_rate": success,
        "mean_p_gs": mean_p,
        "mean_abs_beta": mean_b,
        "median_abs_beta": med_b,
        "mean_max_abs_beta": mean_max_b,
        "frac_over_beta_max": frac,
        "beta_cut_frac": cut,
        "created_utc": data.get("created_utc"),
    }


def write_summary(rows: list[dict], notes: list[str]) -> None:
    lines = [
        "# β-aware fleet summary (ck_pi4)",
        "",
        f"Updated UTC: {_now()}",
        "",
        f"Baseline (`fleet_phase_bakeoff` ck_pi4 L*=4): success={BASELINE_SUCCESS:.2f}, mean|β|≈{BASELINE_MEAN_ABS_BETA:.2f}.",
        "",
        "## Results",
        "",
        "| tag | λ1 | λ3 | β_max | L* | success | mean p(GS) | mean\\|β\\| | median\\|β\\| | mean trial-max\\|β\\| | % over β_max | \\|β\\| cut vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        bm = r["beta_max"]
        bm_s = "—" if bm is None else f"{bm:g}"
        frac = r["frac_over_beta_max"]
        frac_s = "—" if frac != frac else f"{100*frac:.1f}%"
        cut = r["beta_cut_frac"]
        cut_s = "—" if cut != cut else f"{100*cut:.1f}%"
        lines.append(
            f"| `{r['tag']}` | {r['lambda1']:g} | {r['lambda3']:g} | {bm_s} | {r['layers']} | "
            f"{r['success_rate']:.3f} | {r['mean_p_gs']:.4f} | {r['mean_abs_beta']:.3f} | "
            f"{r['median_abs_beta']:.3f} | {r['mean_max_abs_beta']:.3f} | {frac_s} | {cut_s} |"
        )
    lines += ["", "## Notes", ""]
    lines.extend(f"- {n}" for n in notes)
    lines += [
        "",
        "## How to toggle",
        "",
        "Defaults (`--lambda1 0 --lambda3 0`, no `--beta-max`) preserve Gibbs-only cost.",
        "Enable β terms via CLI on `python -m noiseless.run_u_sweep`.",
        "",
    ]
    _SUMMARY.write_text("\n".join(lines) + "\n")
    log(f"wrote {_SUMMARY}")


def try_git_push(msg: str) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Zach He",
        "GIT_AUTHOR_EMAIL": "zach102824@users.noreply.github.com",
        "GIT_COMMITTER_NAME": "Zach He",
        "GIT_COMMITTER_EMAIL": "zach102824@users.noreply.github.com",
    }
    try:
        subprocess.run(["git", "add", "noiseless/results/BETA_AWARE_SUMMARY.md"], cwd=_REPO, check=False)
        # also add newest fleet json/summary matching fleet_beta_aware*
        for p in _RESULTS.glob("fleet_beta_aware_*"):
            subprocess.run(["git", "add", str(p.relative_to(_REPO))], cwd=_REPO, check=False)
        st = subprocess.run(["git", "status", "--porcelain"], cwd=_REPO, capture_output=True, text=True)
        if not st.stdout.strip():
            log("git: nothing to commit")
            return
        subprocess.run(["git", "commit", "-m", msg], cwd=_REPO, env=env, check=False)
        subprocess.run(["git", "push", "origin", "main"], cwd=_REPO, check=False)
        log("git: pushed")
    except Exception as exc:  # noqa: BLE001
        log(f"git push failed: {exc}")


def main() -> int:
    rows: list[dict] = []
    notes: list[str] = []
    os.chdir(_REPO)

    # --- A0 ---
    p = run_fleet("fleet_beta_aware_A0_ckpi4", lambda1=0.0, lambda3=0.0)
    if p:
        m = load_metrics(p)
        rows.append(m)
        notes.append(f"A0 sanity success={m['success_rate']:.3f} (baseline {BASELINE_SUCCESS}).")
        write_summary(rows, notes)
        try_git_push("β-aware A0 ck_pi4 L*=4 results + summary")
    else:
        notes.append("A0 FAILED to run.")
        write_summary(rows, notes)
        return 1

    # --- A1 denser λ1 grid ---
    a1_lambdas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    good_l1: list[float] = []  # success≥0.90 and β cut ≥15%
    for lam in a1_lambdas:
        tag = f"fleet_beta_aware_A1_l1_{str(lam).replace('.', 'p')}"
        # prettier tags for common values
        tag_map = {
            0.005: "fleet_beta_aware_A1_l1_0p005",
            0.01: "fleet_beta_aware_A1_l1_0p01",
            0.02: "fleet_beta_aware_A1_l1_0p02",
            0.05: "fleet_beta_aware_A1_l1_0p05",
            0.1: "fleet_beta_aware_A1_l1_0p1",
            0.2: "fleet_beta_aware_A1_l1_0p2",
            0.5: "fleet_beta_aware_A1_l1_0p5",
        }
        tag = tag_map[lam]
        p = run_fleet(tag, lambda1=lam, lambda3=0.0)
        if not p:
            notes.append(f"A1 λ1={lam} FAILED.")
            write_summary(rows, notes)
            continue
        m = load_metrics(p)
        rows.append(m)
        cut = m["beta_cut_frac"]
        notes.append(
            f"A1 λ1={lam}: success={m['success_rate']:.3f}, mean|β|={m['mean_abs_beta']:.3f}, cut={100*cut:.1f}%."
        )
        write_summary(rows, notes)
        if m["success_rate"] >= 0.90 and cut == cut and cut >= 0.15:
            good_l1.append(lam)
        if m["success_rate"] < 0.85:
            notes.append(f"Stopping A1 escalation: success {m['success_rate']:.3f} < 0.85 at λ1={lam}.")
            write_summary(rows, notes)
            break
        try_git_push(f"β-aware A1 λ1={lam} results")

    # --- A2 soft-cap if any good λ1 ---
    if good_l1:
        # pick best: largest cut among success≥0.90
        candidates = [r for r in rows if r["lambda1"] in good_l1 and r["lambda3"] == 0.0 and r.get("beta_max") is None]
        candidates.sort(key=lambda r: r["beta_cut_frac"], reverse=True)
        best_l1 = float(candidates[0]["lambda1"]) if candidates else good_l1[-1]
        notes.append(f"A2 using best λ1={best_l1} (and λ1=0) × λ3∈{{0.1,1}} × β_max∈{{2,2.5,3}}.")
        write_summary(rows, notes)
        for lam in sorted({0.0, best_l1}):
            for l3 in (0.1, 1.0):
                for bm in (2.0, 2.5, 3.0):
                    tag = (
                        f"fleet_beta_aware_A2_l1_{str(lam).replace('.', 'p')}"
                        f"_l3_{str(l3).replace('.', 'p')}_bmax_{str(bm).replace('.', 'p')}"
                    )
                    p = run_fleet(tag, lambda1=lam, lambda3=l3, beta_max=bm)
                    if not p:
                        notes.append(f"A2 {tag} FAILED.")
                        continue
                    m = load_metrics(p)
                    rows.append(m)
                    write_summary(rows, notes)
        try_git_push("β-aware A2 soft-cap fleet results")
    else:
        notes.append("No λ1 with success≥0.90 AND |β| cut≥15% → skip A2.")
        write_summary(rows, notes)

    # --- Optional: best setting at L*=3 and/or 400 steps ---
    survivors = [
        r for r in rows
        if r["success_rate"] >= 0.90 and r["beta_cut_frac"] == r["beta_cut_frac"] and r["beta_cut_frac"] >= 0.15
    ]
    if survivors:
        survivors.sort(key=lambda r: (r["beta_cut_frac"], r["success_rate"]), reverse=True)
        best = survivors[0]
        notes.append(f"Optional follow-ups on best: {best['tag']}.")
        # L*=3
        tag3 = best["tag"] + "_L3"
        p = run_fleet(
            tag3,
            lambda1=float(best["lambda1"]),
            lambda3=float(best["lambda3"]),
            beta_max=best["beta_max"],
            layers="3",
        )
        if p:
            rows.append(load_metrics(p))
            write_summary(rows, notes)
        # 400 steps on best λ1 only (prefer pure A1 if possible)
        pure = [r for r in survivors if r["lambda3"] == 0.0 and r.get("beta_max") is None]
        focus = pure[0] if pure else best
        tag400 = f"fleet_beta_aware_best_l1_{str(focus['lambda1']).replace('.', 'p')}_steps400"
        p = run_fleet(tag400, lambda1=float(focus["lambda1"]), lambda3=0.0, steps=400)
        if p:
            rows.append(load_metrics(p))
            write_summary(rows, notes)
        try_git_push("β-aware optional L*=3 / steps400 follow-ups")

    # Final recommendation
    good = [
        r for r in rows
        if r["success_rate"] >= 0.90 and r["beta_cut_frac"] == r["beta_cut_frac"] and r["beta_cut_frac"] >= 0.20
    ]
    ok15 = [
        r for r in rows
        if r["success_rate"] >= 0.90 and r["beta_cut_frac"] == r["beta_cut_frac"] and r["beta_cut_frac"] >= 0.15
    ]
    if good:
        good.sort(key=lambda r: r["beta_cut_frac"], reverse=True)
        b = good[0]
        notes.append(
            f"RECOMMEND (≥20% |β| cut, success≥0.90): λ1={b['lambda1']}, λ3={b['lambda3']}, "
            f"β_max={b['beta_max']} → success={b['success_rate']:.3f}, cut={100*b['beta_cut_frac']:.1f}%."
        )
    elif ok15:
        ok15.sort(key=lambda r: r["beta_cut_frac"], reverse=True)
        b = ok15[0]
        notes.append(
            f"RECOMMEND (best ≥15% cut): λ1={b['lambda1']}, λ3={b['lambda3']}, "
            f"β_max={b['beta_max']} → success={b['success_rate']:.3f}, cut={100*b['beta_cut_frac']:.1f}%."
        )
    else:
        notes.append("No setting met success≥0.90 with ≥15% |β| cut; see table for tradeoffs.")
    write_summary(rows, notes)
    try_git_push("β-aware campaign final summary")
    log("campaign complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

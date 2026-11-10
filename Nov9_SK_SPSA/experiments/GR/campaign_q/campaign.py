"""Durable queue for NOV3_QAOA_QL_v1.

Preregistration is prereg.json. This process does not retune v1.
Dev seeds are 10000-10009 only. Final seeds open after FREEZE_n{N}.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common_campaign import (
    OUT,
    RECIPE,
    allocation,
    atomic_write,
    job_id,
    load_prereg,
    prereg_sha,
    result_path,
    runnable_qs,
    seed_span,
)
from worker import conformance_static, execute

HEADER = [
    "stage", "N", "L", "q", "Q", "planned_paired",
    "n_nov3", "n_qaoa", "n_paired", "n_fail_nov3", "n_fail_qaoa",
    "meanF_nov3", "loF_nov3", "hiF_nov3",
    "meanF_qaoa", "loF_qaoa", "hiF_qaoa",
    "mean_dF", "lo_dF", "hi_dF",
    "winF_nov3", "tieF", "winE_nov3",
    "tail70_nov3", "tail80_nov3", "tail70_qaoa", "tail80_qaoa",
    "mean_Eerr_nov3", "mean_Eerr_qaoa",
    "mean_queries_nov3", "mean_queries_qaoa",
    "mean_seconds_nov3", "mean_seconds_qaoa",
    "mean_dF_vs1024_nov3", "lo_vs1024_nov3", "hi_vs1024_nov3",
    "mean_dF_vs1024_qaoa", "lo_vs1024_qaoa", "hi_vs1024_qaoa",
    "complete_1000",
]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(obj) + "\n")
        handle.flush()


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}"


def mean_ci(values: list[float], z: float = 1.96) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "se": None}
    mean = float(np.mean(values))
    if n < 2:
        return {"n": n, "mean": mean, "lo": None, "hi": None, "se": None}
    se = float(np.std(values, ddof=1)) / (n ** 0.5)
    return {"n": n, "mean": mean, "lo": mean - z * se, "hi": mean + z * se, "se": se}


def build_jobs(stage: str, n: int, prereg: dict, sha: str) -> list[dict]:
    seeds = seed_span(prereg["seeds"]["dev" if stage == "dev" else "final"])
    jobs = []
    for seed in seeds:
        for L in prereg["L"]:
            for q in runnable_qs(L, prereg["budget"]["q_candidates"]):
                for method in prereg["methods"]:
                    jobs.append(
                        {
                            "stage": stage,
                            "seed": int(seed),
                            "N": int(n),
                            "L": int(L),
                            "q": int(q),
                            "method": method,
                            "attempt": 1,
                            "protocol_sha": sha,
                        }
                    )
    return jobs


def write_freeze(n: int, prereg: dict, sha: str) -> None:
    path = OUT / f"FREEZE_n{n}.json"
    grid = {str(L): runnable_qs(L, prereg["budget"]["q_candidates"]) for L in prereg["L"]}
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("protocol_sha") != sha or old.get("retuned") is not False or old.get("grid_q") != grid:
            raise SystemExit(f"FREEZE_n{n}.json does not match v1; refusing to edit it")
        return
    atomic_write(
        path,
        json.dumps(
            {
                "recipe": RECIPE,
                "protocol_sha": sha,
                "N": n,
                "retuned": False,
                "grid_q": grid,
                "dev_seeds": "10000-10009",
                "final_seeds": "20000-20999",
                "created_at": _now(),
                "note": "v1 grid copied from prereg.json. Dev outcomes do not edit it.",
            },
            indent=2,
        )
        + "\n",
    )


class Planner:
    def __init__(self, prereg: dict, sha: str, done: set[str]):
        self.prereg = prereg
        self.sha = sha
        self.done = done
        self.stages = []
        for n in prereg["N"]:
            self.stages.append(("dev", int(n)))
            self.stages.append(("final", int(n)))
        self.si = 0
        self.jobs: list[dict] = []
        self.cursor = 0
        self.finished = False
        self.stage_name = ""
        self.stage_n = None
        self._enter()

    def _enter(self) -> None:
        while self.si < len(self.stages):
            stage, n = self.stages[self.si]
            self.stage_name = stage
            self.stage_n = n
            self.jobs = build_jobs(stage, n, self.prereg, self.sha)
            self.cursor = 0
            if any(job_id(job) not in self.done for job in self.jobs):
                return
            self._finish_stage()
            self.si += 1
        self.finished = True
        self.jobs = []

    def _finish_stage(self) -> None:
        stage, n = self.stages[self.si]
        if stage == "dev":
            write_freeze(n, self.prereg, self.sha)
            print(f"freeze N={n} written", flush=True)

    def next_pending(self) -> dict | None:
        if self.finished:
            return None
        while self.cursor < len(self.jobs):
            job = self.jobs[self.cursor]
            self.cursor += 1
            if job_id(job) not in self.done:
                return job
        if self.jobs and all(job_id(job) in self.done for job in self.jobs):
            self._finish_stage()
            self.si += 1
            self._enter()
            return self.next_pending()
        return None


class Aggregator:
    def __init__(self):
        self.ok: dict[tuple, dict] = {}
        self.fail: dict[tuple, int] = {}
        self.seen: set[str] = set()

    def add(self, rec: dict) -> None:
        jid = str(rec.get("job_id") or "")
        if not jid or jid in self.seen or rec.get("kind") == "noop":
            return
        self.seen.add(jid)
        key = (rec.get("stage"), int(rec["N"]), int(rec["L"]), int(rec["q"]), rec.get("method"))
        if rec.get("status") == "ok":
            self.ok.setdefault(key, {})[int(rec["seed"])] = {
                "F": float(rec["F"]),
                "E": float(rec["E"]),
                "Eerr": float(rec["energy_error"]),
                "seconds": float(rec["seconds"]),
                "queries": int(rec["queries"]["total"]),
            }
        else:
            self.fail[key] = self.fail.get(key, 0) + 1

    def load(self) -> None:
        if not (OUT / "runs").exists():
            return
        for path in (OUT / "runs").rglob("*.json"):
            if path.name.endswith(".tmp"):
                continue
            try:
                self.add(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                print(f"skip unreadable {path}: {exc}", flush=True)


def _cell_values(agg: Aggregator, stage, n, L, q, method) -> dict[int, dict]:
    return agg.ok.get((stage, int(n), int(L), int(q), method), {})


def _rows(agg: Aggregator, prereg: dict) -> list[dict]:
    stages = []
    for n in prereg["N"]:
        stages.append(("dev", int(n), 10))
        stages.append(("final", int(n), 1000))
    rows = []
    for stage, n, planned in stages:
        for L in prereg["L"]:
            qs = runnable_qs(L, prereg["budget"]["q_candidates"])
            for q in qs:
                nov = _cell_values(agg, stage, n, L, q, "Nov3rd")
                qaoa = _cell_values(agg, stage, n, L, q, "QAOA")
                if not nov and not qaoa and agg.fail.get((stage, n, L, q, "Nov3rd"), 0) == 0 and agg.fail.get((stage, n, L, q, "QAOA"), 0) == 0:
                    continue
                paired_seeds = sorted(set(nov) & set(qaoa))
                dF = [nov[s]["F"] - qaoa[s]["F"] for s in paired_seeds]
                dE = [nov[s]["E"] - qaoa[s]["E"] for s in paired_seeds]
                f_nov = mean_ci([row["F"] for row in nov.values()])
                f_q = mean_ci([row["F"] for row in qaoa.values()])
                dF_ci = mean_ci(dF)
                ref_n = _cell_values(agg, stage, n, L, 1024, "Nov3rd")
                ref_q = _cell_values(agg, stage, n, L, 1024, "QAOA")
                vs_n = mean_ci([nov[s]["F"] - ref_n[s]["F"] for s in sorted(set(nov) & set(ref_n))]) if q != 1024 else mean_ci([])
                vs_q = mean_ci([qaoa[s]["F"] - ref_q[s]["F"] for s in sorted(set(qaoa) & set(ref_q))]) if q != 1024 else mean_ci([])
                n_paired = len(paired_seeds)
                n_fail_n = agg.fail.get((stage, n, L, q, "Nov3rd"), 0)
                n_fail_q = agg.fail.get((stage, n, L, q, "QAOA"), 0)
                rows.append(
                    {
                        "stage": stage,
                        "N": n,
                        "L": L,
                        "q": q,
                        "Q": allocation(q, L)["Q"],
                        "planned_paired": planned,
                        "n_nov3": len(nov),
                        "n_qaoa": len(qaoa),
                        "n_paired": n_paired,
                        "n_fail_nov3": n_fail_n,
                        "n_fail_qaoa": n_fail_q,
                        "meanF_nov3": f_nov["mean"],
                        "loF_nov3": f_nov["lo"],
                        "hiF_nov3": f_nov["hi"],
                        "meanF_qaoa": f_q["mean"],
                        "loF_qaoa": f_q["lo"],
                        "hiF_qaoa": f_q["hi"],
                        "mean_dF": dF_ci["mean"],
                        "lo_dF": dF_ci["lo"],
                        "hi_dF": dF_ci["hi"],
                        "winF_nov3": (sum(v > 0.0 for v in dF) / n_paired) if n_paired else None,
                        "tieF": (sum(v == 0.0 for v in dF) / n_paired) if n_paired else None,
                        "winE_nov3": (sum(v < 0.0 for v in dE) / n_paired) if n_paired else None,
                        "tail70_nov3": (sum(row["F"] >= 0.7 for row in nov.values()) / len(nov)) if nov else None,
                        "tail80_nov3": (sum(row["F"] >= 0.8 for row in nov.values()) / len(nov)) if nov else None,
                        "tail70_qaoa": (sum(row["F"] >= 0.7 for row in qaoa.values()) / len(qaoa)) if qaoa else None,
                        "tail80_qaoa": (sum(row["F"] >= 0.8 for row in qaoa.values()) / len(qaoa)) if qaoa else None,
                        "mean_Eerr_nov3": float(np.mean([row["Eerr"] for row in nov.values()])) if nov else None,
                        "mean_Eerr_qaoa": float(np.mean([row["Eerr"] for row in qaoa.values()])) if qaoa else None,
                        "mean_queries_nov3": float(np.mean([row["queries"] for row in nov.values()])) if nov else None,
                        "mean_queries_qaoa": float(np.mean([row["queries"] for row in qaoa.values()])) if qaoa else None,
                        "mean_seconds_nov3": float(np.mean([row["seconds"] for row in nov.values()])) if nov else None,
                        "mean_seconds_qaoa": float(np.mean([row["seconds"] for row in qaoa.values()])) if qaoa else None,
                        "mean_dF_vs1024_nov3": vs_n["mean"],
                        "lo_vs1024_nov3": vs_n["lo"],
                        "hi_vs1024_nov3": vs_n["hi"],
                        "mean_dF_vs1024_qaoa": vs_q["mean"],
                        "lo_vs1024_qaoa": vs_q["lo"],
                        "hi_vs1024_qaoa": vs_q["hi"],
                        "complete_1000": stage == "final" and n_paired == 1000 and n_fail_n == 0 and n_fail_q == 0,
                    }
                )
    return rows


def qstar_lines(rows: list[dict], prereg: dict) -> list[str]:
    lines = []
    taus = prereg["tolerances_mean_F"]
    for stage in ("dev", "final"):
        for n in prereg["N"]:
            for L in prereg["L"]:
                have = [row for row in rows if row["stage"] == stage and row["N"] == n and row["L"] == L]
                if not any(row["n_nov3"] or row["n_qaoa"] for row in have):
                    continue
                qs = runnable_qs(L, prereg["budget"]["q_candidates"])
                bits = []
                for method, col in (("Nov3rd", "mean_dF_vs1024_nov3"), ("QAOA", "mean_dF_vs1024_qaoa")):
                    by_q = {row["q"]: row for row in have}
                    for tau in taus:
                        star = None
                        for q in sorted(qs):
                            higher = [qq for qq in qs if qq >= q]
                            ok = True
                            resolved = True
                            for qq in higher:
                                row = by_q.get(qq)
                                if qq == 1024:
                                    continue
                                if row is None or row[col] is None:
                                    ok = False
                                    break
                                if abs(row[col]) > tau:
                                    ok = False
                                    break
                                lo = row["lo_vs1024_nov3" if method == "Nov3rd" else "lo_vs1024_qaoa"]
                                hi = row["hi_vs1024_nov3" if method == "Nov3rd" else "hi_vs1024_qaoa"]
                                if lo is None or hi is None or (hi - lo) / 2 > tau:
                                    resolved = False
                            if ok:
                                star = q
                                bits.append(f"{method} τ={tau:.2f} q*={star}" + ("" if resolved else " (CI wider than τ)"))
                                break
                        if star is None:
                            bits.append(f"{method} τ={tau:.2f} q*=none")
                if bits:
                    lines.append(f"N={n} L={L} {stage}: " + "; ".join(bits))
    return lines


def write_progress(agg: Aggregator, prereg: dict, status: dict) -> None:
    rows = _rows(agg, prereg)
    lines = [",".join(HEADER)]
    for row in rows:
        lines.append(",".join(_fmt(row[name]) for name in HEADER))
    atomic_write(OUT / "PROGRESS.csv", "\n".join(lines) + "\n")
    md = []
    md.append(f"# {RECIPE}")
    md.append("")
    md.append(f"Protocol sha `{status.get('protocol_sha', '')}`. Updated {status.get('updated_at', '')}.")
    md.append("Partial cells are partial. `complete_1000` is true only when a final cell has 1000 paired ok rows and zero failures. q=1024 is a reference budget, not a converged optimizer. A minimal q is relative to cells that exist, and it does not drop final rows.")
    md.append("Dev seeds are 10000–10009. Seeds 10010–10031 and validation 11000–11031 are not read. Local R4 transfer numbers are not part of this table.")
    md.append("")
    md.append(f"Ok rows {status.get('n_ok', 0)}. Failed rows {status.get('n_failed', 0)}. PID {status.get('pid')}.")
    if status.get("eta"):
        md.append("")
        md.append("## Resource estimate")
        md.append("")
        md.append(status["eta"])
    md.append("")
    md.append("| stage | N | L | q | Q | n paired | mean F Nov3 | 95% CI | mean F QAOA | 95% CI | mean ΔF | 95% CI | F wins | E wins | F≥0.7 Nov3/QAOA | F≥0.8 | mean E−E0 Nov3/QAOA | mean sec |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---|---:|---:|---|---|---|---:|")
    for row in rows:
        def ci(lo, hi):
            if lo is None:
                return ""
            return f"[{lo:.3f}, {hi:.3f}]"

        def num(value, nd=3):
            return "" if value is None else f"{value:.{nd}f}"

        md.append(
            "| {stage} | {N} | {L} | {q} | {Q} | {n}/{plan} | {fn} | {cin} | {fq} | {ciq} | {df} | {cid} | {wf} | {we} | {t7} | {t8} | {ee} | {sec} |".format(
                stage=row["stage"],
                N=row["N"],
                L=row["L"],
                q=row["q"],
                Q=row["Q"],
                n=row["n_paired"],
                plan=row["planned_paired"],
                fn=num(row["meanF_nov3"]),
                cin=ci(row["loF_nov3"], row["hiF_nov3"]),
                fq=num(row["meanF_qaoa"]),
                ciq=ci(row["loF_qaoa"], row["hiF_qaoa"]),
                df=num(row["mean_dF"]),
                cid=ci(row["lo_dF"], row["hi_dF"]),
                wf=num(row["winF_nov3"], 2),
                we=num(row["winE_nov3"], 2),
                t7=f"{num(row['tail70_nov3'], 2)}/{num(row['tail70_qaoa'], 2)}",
                t8=f"{num(row['tail80_nov3'], 2)}/{num(row['tail80_qaoa'], 2)}",
                ee=f"{num(row['mean_Eerr_nov3'])}/{num(row['mean_Eerr_qaoa'])}",
                sec=num(row["mean_seconds_nov3"], 2),
            )
        )
    stars = qstar_lines(rows, prereg)
    if stars:
        md.append("")
        md.append("## Relative minimal q")
        md.append("")
        md.append("q* is the smallest runnable q such that every larger runnable q is within τ in mean fidelity of the q=1024 reference of the same stage, N, L, and method. A wider interval than τ means the tolerance is not resolved.")
        md.append("")
        for line in stars:
            md.append(f"- {line}")
    md.append("")
    atomic_write(OUT / "SUMMARY.md", "\n".join(md) + "\n")


def measure_scaling(prereg: dict) -> dict:
    from hamiltonian import make_J, pair_diag, z_table
    from r2_b3 import energy_of, features, n_controls, nov3_psi, qaoa_psi

    rows = []
    for n in prereg["N"]:
        for L in (1, 6):
            for method in prereg["methods"]:
                t0 = time.perf_counter()
                try:
                    J = make_J(10000, int(n))
                    hc = pair_diag(J, z_table(int(n)))
                    groups, dk, dm, hc = features(J, int(n), hc)
                    x = np.zeros(n_controls(method, int(L)))

                    def one(x=x, method=method, L=L, groups=groups, dk=dk, dm=dm, hc=hc, n=n):
                        if method == "Nov3rd":
                            psi = nov3_psi(x, int(L), hc, dk, dm, groups, int(n))
                        else:
                            psi = qaoa_psi(x, int(L), hc, int(n))
                        return energy_of(psi, hc, 0.5, int(n))

                    one()
                    repeats = 3
                    t1 = time.perf_counter()
                    for _ in range(repeats):
                        one()
                    dt = (time.perf_counter() - t1) / repeats
                    row = {"N": int(n), "L": int(L), "method": method, "seconds_per_eval": dt, "b": 0.5, "label": "throughput_not_Bstar_not_a_panel_fit", "error": None}
                except Exception as exc:
                    row = {"N": int(n), "L": int(L), "method": method, "seconds_per_eval": None, "error": str(exc), "elapsed": time.perf_counter() - t0}
                rows.append(row)
                print(f"scale N={n} L={L} {method} {row.get('seconds_per_eval')}", flush=True)
    doc = {"protocol_sha": prereg["_sha256"], "rows": rows, "created_at": _now()}
    atomic_write(OUT / "scaling.json", json.dumps(doc, indent=2) + "\n")
    return doc


def _scale_table(doc: dict) -> dict:
    table = {}
    for row in doc.get("rows", []):
        if row.get("seconds_per_eval"):
            table[(int(row["N"]), int(row["L"]), row["method"])] = float(row["seconds_per_eval"])
    return table


def eta_text(prereg: dict, agg: Aggregator, scale_doc: dict, workers: int) -> str:
    table = _scale_table(scale_doc)
    lines = ["Throughput below is seconds per energy evaluation at b=0.5, not a B* optimization. Panel ETA multiplies that by Q=qL and divides by the worker count. Observed SPSA seconds replace the probe once a size has two finished runs."]
    for n in prereg["N"]:
        bits = []
        for L in (1, 6):
            for method in prereg["methods"]:
                value = table.get((int(n), int(L), method))
                if value is not None:
                    bits.append(f"L{L} {method} {value:.4f}s")
        samples = []
        for key, seeds in agg.ok.items():
            if key[1] == int(n):
                samples.extend(row["seconds"] / row["queries"] for row in seeds.values() if row["queries"])
        observed = float(np.median(samples)) if len(samples) >= 2 else None
        if observed is not None:
            bits.append(f"observed median {observed:.4f}s/query from {len(samples)} runs")
        lines.append(f"N={n}: " + ", ".join(bits))
    stored = {int(n): 0 for n in prereg["N"]}
    done_q = 0
    for key, seeds in agg.ok.items():
        charged = sum(row["queries"] for row in seeds.values())
        stored[int(key[1])] = stored.get(int(key[1]), 0) + charged
        done_q += charged

    def sec_eval(n: int, L: int, method: str):
        s1 = table.get((n, 1, method))
        s6 = table.get((n, 6, method))
        if s1 is None or s6 is None:
            return None
        return s1 + (L - 1) / 5.0 * (s6 - s1)

    total_sec = 0.0
    for n in prereg["N"]:
        planned = 0
        weighted = 0.0
        known = True
        for nseed in (10, 1000):
            for L in prereg["L"]:
                for q in runnable_qs(L, prereg["budget"]["q_candidates"]):
                    queries = allocation(q, L)["Q"] * 2 * nseed
                    planned += queries
                    for method in prereg["methods"]:
                        rate = sec_eval(int(n), int(L), method)
                        if rate is None:
                            known = False
                        else:
                            weighted += allocation(q, L)["Q"] * nseed * rate
        left_queries = max(0, planned - stored.get(int(n), 0))
        if not known or planned <= 0:
            lines.append(f"N={n} remaining queries {left_queries}, rate unknown")
            continue
        secs = (weighted / max(1, workers)) * (left_queries / planned)
        total_sec += secs
        lines.append(
            f"N={n} planned queries {planned}, remaining {left_queries}, about {secs/3600:.2f} h on {workers} workers at the L-interpolated b=0.5 probe"
        )
    lines.append(f"Rough sum across N, if those probe rates hold: {total_sec/3600:.1f} h. Not a finished panel.")
    lines.append(f"Charged ok queries so far {done_q}.")
    return "\n".join(lines)


def replay_report() -> dict:
    from common import RESULT_DIR

    report = {"cells": {}, "pass": True}
    for method in ("Nov3rd", "QAOA"):
        ours_path = result_path({"N": 6, "seed": 10000, "L": 1, "q": 1024, "method": method})
        ref_path = RESULT_DIR / "r2_b3" / "runs" / f"n6_s10000_L1_{method}_c1024.json"
        ours = json.loads(ours_path.read_text())
        ref = json.loads(ref_path.read_text())
        if ours.get("status") != "ok":
            report["cells"][method] = {"pass": False, "status": ours.get("status"), "reason": ours.get("reason")}
            report["pass"] = False
            continue
        dE = abs(float(ours["E"]) - float(ref["E"]))
        dF = abs(float(ours["F"]) - float(ref["F"]))
        dG = max(abs(float(a) - float(b)) for a, b in zip(ours["a_gains"], ref["a_gains"]))
        ok = dE <= 1e-8 and dG <= 1e-9 and ours["queries"]["total"] == 1024
        report["cells"][method] = {"dE": dE, "dF": dF, "dGain": dG, "pass": ok, "E": ours["E"], "F": ours["F"]}
        report["pass"] = report["pass"] and ok
    return report


def collect_done() -> set[str]:
    done = set()
    if not (OUT / "runs").exists():
        return done
    for path in (OUT / "runs").rglob("*.json"):
        if path.name.endswith(".tmp"):
            continue
        # nXX/sSEED/Lq_method.json -> job id uses unpadded N
        try:
            rec = json.loads(path.read_text())
            if rec.get("job_id"):
                done.add(rec["job_id"])
        except (OSError, json.JSONDecodeError):
            continue
    return done


def write_status(payload: dict) -> None:
    payload["updated_at"] = _now()
    atomic_write(OUT / "STATUS.json", json.dumps(payload, indent=2) + "\n")


def take_pid() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "campaign.pid"
    if path.exists():
        try:
            pid = int(path.read_text().strip() or "0")
        except ValueError:
            pid = 0
        if pid and _pid_alive(pid) and pid != os.getpid():
            raise SystemExit(f"campaign already running as pid {pid}")
    path.write_text(str(os.getpid()))


def handle_record(job: dict, rec: dict, agg: Aggregator, done: set[str]) -> dict | None:
    attempt = int(job.get("attempt", 1))
    if rec.get("status") != "ok" and attempt < 2 and rec.get("retryable", True):
        retry = dict(job)
        retry["attempt"] = attempt + 1
        _append(OUT / "events.jsonl", {"ts": _now(), "event": "retry", "job_id": job_id(job), "attempt": 2})
        return retry
    path = result_path(job)
    rec["job_id"] = job_id(job)
    rec.setdefault("stage", job["stage"])
    rec.setdefault("seed", job["seed"])
    rec.setdefault("N", job["N"])
    rec.setdefault("L", job["L"])
    rec.setdefault("q", job["q"])
    rec.setdefault("method", job["method"])
    rec["written_at"] = _now()
    atomic_write(path, json.dumps(rec, indent=2) + "\n")
    done.add(rec["job_id"])
    agg.add(rec)
    event = {"ts": _now(), "event": rec.get("status"), "job_id": rec["job_id"], "F": rec.get("F"), "E": rec.get("E"), "seconds": rec.get("seconds")}
    _append(OUT / "events.jsonl", event)
    if rec.get("status") != "ok":
        _append(OUT / "failures.jsonl", rec)
    ftxt = f" F={rec['F']:.4f}" if rec.get("status") == "ok" else " FAIL"
    print(f"done {rec['job_id']} {rec.get('status')}{ftxt} q={rec.get('queries', {}).get('total')} {rec.get('seconds')}", flush=True)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    prereg = load_prereg()
    sha = prereg["_sha256"]
    if args.self_test:
        report = conformance_static()
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(2) as pool:
            out = pool.map(execute, [{"kind": "noop", "echo": 1}, {"kind": "noop", "echo": 2}])
        if [row.get("echo") for row in out] != [1, 2]:
            raise SystemExit(f"spawn pool failed {out}")
        print(json.dumps({"passed": report["passed"], "n_checks": len(report["checks"]), "spawn": True}))
        return

    OUT.mkdir(parents=True, exist_ok=True)
    sha_path = OUT / "protocol_sha.txt"
    if sha_path.exists() and sha_path.read_text().strip() != sha:
        raise SystemExit("prereg.json changed after the campaign started; v1 is immutable")
    sha_path.write_text(sha + "\n")
    take_pid()
    static_path = OUT / "conformance_static.json"
    if not static_path.exists():
        atomic_write(static_path, json.dumps(conformance_static(), indent=2) + "\n")
        print("static conformance passed", flush=True)
    scale_path = OUT / "scaling.json"
    if not scale_path.exists():
        scale_doc = measure_scaling(prereg)
    else:
        scale_doc = json.loads(scale_path.read_text())
    agg = Aggregator()
    agg.load()
    done = collect_done()
    workers = int(prereg["workers"])
    status = {
        "recipe": RECIPE,
        "protocol_sha": sha,
        "pid": os.getpid(),
        "started_at": _now(),
        "phase": "starting",
        "n_ok": len(agg.seen) - sum(agg.fail.values()),
        "n_failed": sum(agg.fail.values()),
        "workers": workers,
    }
    write_status(status)

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(workers)
    inflight = []
    try:
        replay_ids = {
            job_id({"N": 6, "seed": 10000, "L": 1, "q": 1024, "method": "Nov3rd"}),
            job_id({"N": 6, "seed": 10000, "L": 1, "q": 1024, "method": "QAOA"}),
        }
        if not replay_ids <= done:
            for method in ("Nov3rd", "QAOA"):
                jid = job_id({"N": 6, "seed": 10000, "L": 1, "q": 1024, "method": method})
                if jid in done:
                    continue
                job = {"stage": "dev", "seed": 10000, "N": 6, "L": 1, "q": 1024, "method": method, "attempt": 1, "protocol_sha": sha}
                inflight.append((job, pool.apply_async(execute, (job,))))
            while inflight:
                still = []
                for job, async_result in inflight:
                    if not async_result.ready():
                        still.append((job, async_result))
                        continue
                    rec = async_result.get()
                    retry = handle_record(job, rec, agg, done)
                    if retry is not None:
                        still.append((retry, pool.apply_async(execute, (retry,))))
                inflight = still
                time.sleep(0.2)
        report = replay_report()
        atomic_write(OUT / "conformance_replay.json", json.dumps(report, indent=2) + "\n")
        print("replay", json.dumps(report["cells"]), flush=True)
        if not report["pass"]:
            status["phase"] = "replay_failed"
            status["eta"] = "stopped: N3X replay disagreed"
            write_status(status)
            raise SystemExit("replay gate failed")

        planner = Planner(prereg, sha, done)
        last_write = 0.0
        while not planner.finished or inflight:
            while len(inflight) < workers:
                job = planner.next_pending()
                if job is None:
                    break
                inflight.append((job, pool.apply_async(execute, (job,))))
                _append(OUT / "events.jsonl", {"ts": _now(), "event": "start", "job_id": job_id(job)})
            if not inflight and planner.finished:
                break
            time.sleep(0.2)
            still = []
            progress = False
            for job, async_result in inflight:
                if not async_result.ready():
                    still.append((job, async_result))
                    continue
                progress = True
                try:
                    rec = async_result.get()
                except Exception:
                    rec = {"status": "failed", "retryable": True, "traceback": traceback.format_exc()}
                retry = handle_record(job, rec, agg, done)
                if retry is not None:
                    still.append((retry, pool.apply_async(execute, (retry,))))
            inflight = still
            now = time.time()
            if progress or now - last_write > 30:
                status["phase"] = f"{planner.stage_name} N={planner.stage_n}"
                status["n_ok"] = sum(len(seeds) for seeds in agg.ok.values())
                status["n_failed"] = sum(agg.fail.values())
                status["inflight"] = [job_id(job) for job, _ar in inflight]
                status["pid"] = os.getpid()
                try:
                    status["eta"] = eta_text(prereg, agg, scale_doc, workers)
                except Exception as exc:
                    status["eta"] = f"eta unavailable: {exc}"
                write_status(status)
                try:
                    write_progress(agg, prereg, status)
                except Exception:
                    traceback.print_exc()
                last_write = now
        status["phase"] = "complete" if planner.finished else status.get("phase")
        status["n_ok"] = sum(len(seeds) for seeds in agg.ok.values())
        status["n_failed"] = sum(agg.fail.values())
        status["inflight"] = []
        status["eta"] = eta_text(prereg, agg, scale_doc, workers)
        write_status(status)
        write_progress(agg, prereg, status)
        print("campaign loop ended", status["phase"], flush=True)
    finally:
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

"""Recipe audit for reusing PR82 S4. Mismatch => refit, never free-count."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "results" / "STARTS" / "runs" / "N15"
SEEDS = list(range(10048, 10056))
CIRCUITS = ("Nov3", "QAOA")
NEED = {
    "N": 15, "L": 3, "arm": "S4", "nstarts": 4, "q": 1024, "Q": 3072,
    "status": "ok", "optimizer": "C0", "engine": "m2", "queries": 3072,
    "K_per_start": 379, "per_start": 768, "outside_cap_queries": 0,
    "cap_inclusive": True,
}
CAT = {"calibration": 32, "perturb": 3032, "candidate": 8, "monitor": 0}


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_one(path, seed, circ):
    problems = []
    if not path.exists():
        return ["missing"], None, None
    rec = json.loads(path.read_text())
    for k, v in NEED.items():
        if rec.get(k) != v:
            problems.append(f"{k}={rec.get(k)}")
    if rec.get("circuit") != circ:
        problems.append("circuit")
    if rec.get("seed") != seed:
        problems.append("seed")
    if rec.get("query_cat") != CAT:
        problems.append("query_cat")
    if any(m.get("gain_frozen") for m in rec.get("start_meta") or []):
        problems.append("gain_frozen")
    if rec.get("n_cands") != 8:
        problems.append("n_cands")
    return problems, rec, file_sha(path)


def audit_all():
    rows = []
    for seed in SEEDS:
        for circ in CIRCUITS:
            path = RUNS / f"s{seed}_L3_{circ}_S4_q1024.json"
            problems, rec, sha = audit_one(path, seed, circ)
            rows.append({
                "seed": seed, "circuit": circ, "path": str(path),
                "ok": not problems, "problems": problems, "sha256": sha,
                "prereg_sha256": None if rec is None else rec.get("prereg_sha256"),
            })
    return rows


def all_ok(rows=None):
    rows = audit_all() if rows is None else rows
    return all(r["ok"] for r in rows)

#!/usr/bin/env python3
"""ECD (jp) ansatz on the F1 scaling instances with k bits per cavity (n = 2 + 2k; n = 16 -> k = 7).

Default recipe = tuned n = 8 recipe (GROW_ADAM_TUNING_SUMMARY.md): u = jp, binary encoding, λ1 = λ = 0
Gibbs cost with the sampled-tail η controller on EXACT Born probabilities (infinite shots),
optimizer spsa_adam, layer growth from L = 1 (transparent last layer + kick σ = 0.05), c = 0.15,
per-stage steps / Adam lr schedules (``--steps-schedule`` / ``--lr-schedule``, one entry per stage).

Layout / truncation / leakage: see noiseless/ecd_kbit.py. Each cavity is simulated with ``--nf``
Fock levels (default 160 > 2^7 = 128); every finished trial is also re-evaluated at ``--nf-check``
levels (default 224) to monitor truncation convergence.

Seeds follow noiseless/run_u_sweep.py: seed0 + 100_000*inst + 1_000*U_NAMES.index('jp') + 10*L + trial.

Checkpoint: one JSON line per finished trial in
  noiseless/results/ecd_kbit_runs/<tag>/ecd_jp_L<L>_n<nn>.jsonl   (gitignored), skipped on restart.
``--summarize`` writes noiseless/results/ecd_kbit_<tag>_summary.json.
"""

from __future__ import annotations

import os as _os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from noiseless.ecd_kbit import KbitEcdSimulator, KbitLayout, logical_energies_from_npz  # noqa: E402
from noiseless.spsa_gibbs import GROW_KICK_SIGMA, betas_from_x, grow_trial  # noqa: E402
from noiseless.unitaries import U_NAMES  # noqa: E402

RUN_ROOT = _REPO / "noiseless" / "results" / "ecd_kbit_runs"
HAM_ROOT = _REPO / "Hamiltonians" / "four_sat_scaling"


def instance_paths(n: int) -> list[Path]:
    return sorted((HAM_ROOT / f"n{n:02d}").glob(f"four_sat_n{n:02d}_[0-9][0-9][0-9].npz"))


@lru_cache(maxsize=2)
def _energies(path: str, n: int):
    E, meta = logical_energies_from_npz(path, n)
    gs = meta["ground_bitstring"]
    if int(np.argmin(E)) != int(gs, 2) or np.count_nonzero(np.isclose(E, E.min())) != 1:
        raise ValueError(f"{path}: ground bitstring is not the unique argmin")
    return E, gs


def trial_seed(seed0: int, inst: int, L: int, trial: int, u_name: str = "jp") -> int:
    return int(seed0) + 100_000 * int(inst) + 1_000 * list(U_NAMES).index(u_name) + 10 * int(L) + int(trial)


def make_sim(path: str, n: int, nf: int, L: int, method: str = "fast") -> KbitEcdSimulator:
    E, gs = _energies(path, n)
    k = (n - 2) // 2
    return KbitEcdSimulator(KbitLayout(k, nf), E, L, gs, method=method)


def worker(job: dict) -> dict:
    t0 = time.perf_counter()
    n, L, nf = int(job["n"]), int(job["L"]), int(job["nf"])
    sim = make_sim(job["path"], n, nf, 1)
    rng = np.random.default_rng(job["seed"])
    res = grow_trial(
        sim, final_layers=L, rng=rng, start_layers=1, kick_sigma=job["kick"], c=job["c"],
        optimizer="spsa_adam", steps_schedule=job["steps_schedule"], lr_schedule=job["lr_schedule"],
    )
    t_opt = time.perf_counter() - t0
    x = np.asarray(res.x, dtype=float)
    ev = sim.evaluate(x)
    rec = {k: job[k] for k in ("n", "L", "inst", "trial", "seed", "nf", "steps_schedule", "lr_schedule")}
    rec.update(
        file=Path(job["path"]).name, arm=f"ecd_jp_L{L}", n_params=int(x.size),
        success=bool(ev["success"]), p_gs=ev["p_gs"], most_likely_bitstring=ev["most_likely_bitstring"],
        ground_bitstring=sim.ground_bitstring, energy_mean=ev["energy_mean"],
        energy_mean_valid=ev["energy_mean_valid"], leakage=ev["leakage"], p_top_levels=ev["p_top_levels"],
        mean_n_a=ev["mean_n_a"], mean_n_b=ev["mean_n_b"], mean_abs_beta=ev["mean_abs_beta"],
        max_abs_beta=ev["max_abs_beta"], abs_betas=np.abs(betas_from_x(x, L)).tolist(),
        fun=float(res.fun), eta=float(res.eta), nfev=int(res.nfev), x=x.tolist(), stages=res.stages,
        wall_opt_s=t_opt,
    )
    nfc = int(job.get("nf_check") or 0)
    if nfc:
        evc = make_sim(job["path"], n, nfc, L).evaluate(x)
        rec.update(nf_check=nfc, p_gs_check=evc["p_gs"], leakage_check=evc["leakage"],
                   success_check=bool(evc["success"]),
                   max_abs_dp_check=float(np.abs(
                       evc["probs"].reshape(4, nfc, nfc)[:, :nf, :nf].reshape(-1)
                       - ev["probs"]).max()))
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def ckpt_path(tag: str, L: int, n: int) -> Path:
    return RUN_ROOT / tag / f"ecd_jp_L{L}_n{n:02d}.jsonl"


def _done(tag, L, n) -> set:
    p = ckpt_path(tag, L, n)
    out = set()
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
                out.add((int(r["inst"]), int(r["trial"])))
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def _parse_sched(s, cast, L):
    v = [cast(t) for t in s.split(",") if t.strip()]
    if len(v) != L:
        raise SystemExit(f"schedule {s!r} needs {L} entries (growth 1->{L})")
    return v


def build_jobs(a) -> list[dict]:
    n, L = int(a.n), int(a.layers)
    paths = instance_paths(n)
    insts = list(range(len(paths)))[: a.max_h] if a.instances is None else [int(i) for i in a.instances.split(",")]
    steps = _parse_sched(a.steps_schedule, int, L)
    lrs = _parse_sched(a.lr_schedule, float, L)
    done = _done(a.tag, L, n)
    jobs = []
    for i in insts:
        for t in range(a.trial_offset, a.trial_offset + a.trials):
            if (i, t) in done:
                continue
            jobs.append(dict(n=n, L=L, inst=i, trial=t, path=str(paths[i]), nf=a.nf, nf_check=a.nf_check,
                             seed=trial_seed(a.seed, i, L, t), steps_schedule=steps, lr_schedule=lrs,
                             kick=a.kick, c=a.c))
    return jobs


def summarize(tag: str) -> dict:
    cells = {}
    for p in sorted((RUN_ROOT / tag).glob("*.jsonl")):
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not recs:
            continue
        pg = np.array([r["p_gs"] for r in recs])
        lk = np.array([r["leakage"] for r in recs])
        mb = np.array([r["max_abs_beta"] for r in recs])
        cells[p.stem] = {
            "arm": recs[0]["arm"], "n": recs[0]["n"], "L": recs[0]["L"], "n_params": recs[0]["n_params"],
            "nfev": recs[0]["nfev"], "nf": recs[0]["nf"], "trials_total": len(recs),
            "instances": len({r["inst"] for r in recs}),
            "success_rate": float(np.mean([r["success"] for r in recs])),
            "mean_p_gs": float(pg.mean()), "median_p_gs": float(np.median(pg)),
            "leakage_mean": float(lk.mean()), "leakage_median": float(np.median(lk)), "leakage_max": float(lk.max()),
            "frac_leakage_gt_1e-3": float(np.mean(lk > 1e-3)),
            "max_abs_beta_mean": float(mb.mean()), "max_abs_beta_median": float(np.median(mb)),
            "max_abs_beta_max": float(mb.max()),
            "wall_s_mean_per_trial": float(np.mean([r["wall_s"] for r in recs])),
        }
        if "p_gs_check" in recs[0]:
            cells[p.stem]["max_abs_dp_gs_check"] = float(max(abs(r["p_gs_check"] - r["p_gs"]) for r in recs))
            cells[p.stem]["max_abs_dp_check"] = float(max(r["max_abs_dp_check"] for r in recs))
            cells[p.stem]["success_check_agree"] = float(np.mean([r["success_check"] == r["success"] for r in recs]))
    out = {"tag": tag, "cells": cells}
    (_REPO / "noiseless" / "results" / f"ecd_kbit_{tag}_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--layers", type=int, default=4, help="final depth L (growth 1 -> L)")
    p.add_argument("--steps-schedule", default="200,200,200,200")
    p.add_argument("--lr-schedule", default="0.5,0.2,0.05,0.02")
    p.add_argument("--nf", type=int, default=160)
    p.add_argument("--nf-check", type=int, default=224, help="0 = no re-evaluation")
    p.add_argument("--max-h", type=int, default=20)
    p.add_argument("--instances", default=None)
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--trial-offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--kick", type=float, default=GROW_KICK_SIGMA)
    p.add_argument("--c", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tag", default="main")
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args(argv)
    if a.summarize:
        print(json.dumps(summarize(a.tag), indent=1))
        return 0
    jobs = build_jobs(a)
    ckpt_path(a.tag, a.layers, a.n).parent.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {len(jobs)} jobs tag={a.tag} L={a.layers} nf={a.nf}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(worker, j): j for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001  (re-run on restart)
                print(f"FAILED inst={j['inst']} t={j['trial']}: {e!r}", flush=True)
                continue
            with open(ckpt_path(a.tag, a.layers, a.n), "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(f"[{time.strftime('%H:%M:%S')}] {k}/{len(jobs)} inst={r['inst']} t={r['trial']} "
                  f"succ={r['success']} p_gs={r['p_gs']:.4g} leak={r['leakage']:.2e} "
                  f"max|b|={r['max_abs_beta']:.2f} wall={r['wall_s']:.1f}s elapsed={time.time()-t0:.0f}s", flush=True)
    summarize(a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

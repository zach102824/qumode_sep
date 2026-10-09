#!/usr/bin/env python3
"""Relayout (cavity-XOR relabel) sweep on the k-bit ECD chip for any even n (n = 2 + 2k).

One config = (n, L, s, r, R) [+ lr, nf]: round-0 growth L = 1 -> L with s SPSA-Adam steps per stage,
then R relabel rounds of r steps at L from small beta (lr 0.1), best round by common-eta Gibbs cost.
``R = 0`` with large s is the plain tuned-growth baseline (no relabel).

Hamiltonians: ``--ham-set scaling`` = Hamiltonians/four_sat_scaling/n<nn> (20 planted unique 4-SAT);
``--ham-set legacy8`` = the n = 8 headline set Hamiltonians/four_sat (energies via the legacy n = 8
arithmetic, for the bit-for-bit reproduction).

Seeds (paired across every config, = the headline's --seed-layers 4 formula):
    seed0 + 100_000 * inst + 1_000 * U_NAMES.index('jp') + 10 * 4 + trial.

Checkpoint: one JSON line per trial in ``--outdir/<tag>.jsonl`` (gitignored); finished (inst, trial)
pairs are skipped on restart, so a 25-trial confirm reuses the 5-trial screen's trials 0..4.
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

from noiseless.ecd_kbit import logical_energies_from_npz  # noqa: E402
from noiseless.relayout_kbit import XorKbitSim, KbitLayout, default_nf, explore_exploit_trial_kbit, relayout_trial_kbit  # noqa: E402
from noiseless.unitaries import U_NAMES  # noqa: E402

RUN_ROOT = _REPO / "noiseless" / "results" / "relayout_runs" / "kbit"
HEADLINE_LR = (0.5, 0.2, 0.05, 0.02)


def lr_schedule_for(L: int) -> list[float]:
    """Round-0 growth lr per stage. L = 4: the tuned 0.5/0.2/0.05/0.02. Other L: log-linear
    interpolation of that 4-point schedule over L stages (first 0.5, last 0.02)."""
    L = int(L)
    if L == 4:
        return list(HEADLINE_LR)
    if L == 1:
        return [HEADLINE_LR[0]]
    pos = np.linspace(0.0, 3.0, L)
    return [float(np.exp(np.interp(p, np.arange(4), np.log(HEADLINE_LR)))) for p in pos]


def ham_paths(n: int, ham_set: str) -> list[Path]:
    if ham_set == "legacy8":
        if n != 8:
            raise SystemExit("legacy8 set is n = 8")
        return sorted((_REPO / "Hamiltonians" / "four_sat").glob("four_sat_[0-9][0-9][0-9].npz"))
    return sorted((_REPO / "Hamiltonians" / "four_sat_scaling" / f"n{n:02d}").glob(f"four_sat_n{n:02d}_[0-9][0-9][0-9].npz"))


@lru_cache(maxsize=4)
def energies_for(path: str, n: int, ham_set: str):
    if ham_set == "legacy8":
        from noiseless.encoding import load_four_sat_npz, logical_energies_from_terms

        inst = load_four_sat_npz(path)
        return logical_energies_from_terms(inst["terms"], inst["identity"]), inst["ground_bitstring"]
    E, meta = logical_energies_from_npz(path, n)
    gs = meta["ground_bitstring"]
    if int(np.argmin(E)) != int(gs, 2) or np.count_nonzero(np.isclose(E, E.min())) != 1:
        raise ValueError(f"{path}: ground bitstring is not the unique argmin")
    return E, gs


def trial_seed(seed0: int, inst: int, trial: int) -> int:
    return int(seed0) + 100_000 * int(inst) + 1_000 * list(U_NAMES).index("jp") + 10 * 4 + int(trial)


def config_tag(n, L, s, r, R, lr=0.1, nf=None, ham_set="scaling", method="fast", explore=1, topk=1,
               explore_mask="random", xeta=1.0, reta=1.0, xL=None, xs=None) -> str:
    t = f"n{n:02d}_L{L}s{s}_r{r}_R{R}"
    if int(explore) > 1 or int(topk) > 1:
        t += f"_E{int(explore)}K{int(topk)}" + ("" if explore_mask == "random" else f"{explore_mask}")
        if xeta and float(xeta) != 1.0:
            t += f"_xeta{float(xeta):g}"
        if reta and float(reta) != 1.0:
            t += f"_reta{float(reta):g}"
        if xL:
            t += f"_xL{int(xL)}"
        if xs:
            t += f"_xs{int(xs)}"
    if abs(float(lr) - 0.1) > 1e-12:
        t += f"_lr{lr:g}"
    if nf is not None and nf != default_nf((n - 2) // 2):
        t += f"_nf{nf}"
    if ham_set != "scaling":
        t += f"_{ham_set}"
    if method != "fast":
        t += f"_{method}"
    return t


def worker(job: dict) -> dict:
    t0 = time.perf_counter()
    n = int(job["n"])
    k = (n - 2) // 2
    E, gs = energies_for(job["path"], n, job["ham_set"])
    rng = np.random.default_rng(int(job["seed"]))
    M = int(job.get("explore") or 1)
    K = int(job.get("topk") or 1)
    if M > 1 or K > 1:
        rec = explore_exploit_trial_kbit(
            E, gs, k=k, nf=int(job["nf"]), final_layers=int(job["L"]), rng=rng, explore_rounds=M,
            explore_mask=job.get("explore_mask") or "random", topk=K,
            explore_eta_scale=float(job.get("xeta") or 1.0), exploit_eta_scale=float(job.get("reta") or 1.0),
            explore_layers=job.get("xL"), explore_steps=job.get("xs"),
            relayout_rounds=int(job["R"]), relayout_steps=int(job["r"]), relayout_lr=float(job["lr"]),
            steps_per_stage=int(job["s"]), lr_schedule=lr_schedule_for(int(job["L"])), method=job["method"],
        )
    else:
        rec = relayout_trial_kbit(
            E, gs, k=k, nf=int(job["nf"]), final_layers=int(job["L"]), rng=rng,
            relayout_rounds=int(job["R"]), relayout_steps=int(job["r"]), relayout_lr=float(job["lr"]),
            steps_per_stage=int(job["s"]), lr_schedule=lr_schedule_for(int(job["L"])), method=job["method"],
        )
    rec.update({kk: job[kk] for kk in ("n", "L", "s", "r", "R", "lr", "nf", "inst", "trial", "seed", "ham_set", "method")})
    rec.update(explore=M, topk=K, explore_mask=job.get("explore_mask") or "random",
               xeta=job.get("xeta"), reta=job.get("reta"), xL=job.get("xL"), xs=job.get("xs"))
    rec["file"] = Path(job["path"]).name
    nfc = int(job.get("nf_check") or 0)
    if nfc:
        L = int(job["L"])
        x = np.asarray(rec["x"], dtype=float)
        a = XorKbitSim(KbitLayout(k, int(job["nf"])), E, L, gs, method=job["method"], xa=rec["sel_xa"], xb=rec["sel_xb"]).evaluate(x)
        b = XorKbitSim(KbitLayout(k, nfc), E, L, gs, method="fast", xa=rec["sel_xa"], xb=rec["sel_xb"]).evaluate(x)
        nf = int(job["nf"])
        pb = b["probs"].reshape(4, nfc, nfc)[:, :nf, :nf].reshape(-1)
        rec.update(nf_check=nfc, p_gs_check=float(b["p_gs"]), success_check=bool(b["success"]),
                   leakage_check=float(b["leakage"]), max_abs_dp_check=float(np.abs(pb - a["probs"]).max()))
    rec["wall_s"] = time.perf_counter() - t0
    return rec


def ckpt(outdir: Path, tag: str) -> Path:
    return outdir / f"{tag}.jsonl"


def done_pairs(path: Path) -> set:
    out = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                out.add((int(r["inst"]), int(r["trial"])))
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def load_records(path: Path, trials: int | None = None) -> list[dict]:
    if not path.exists():
        return []
    recs = []
    seen = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (int(r["inst"]), int(r["trial"]))
        if key in seen or (trials is not None and int(r["trial"]) >= trials):
            continue
        seen.add(key)
        recs.append(r)
    return recs


def run_config(*, n, L, s, r, R, lr=0.1, nf=None, trials=5, trial_offset=0, ham_set="scaling", method="fast",
               workers=8, seed0=20260917, outdir: Path = RUN_ROOT, nf_check=None, max_h=None, log=None,
               tag=None, explore=1, topk=1, explore_mask="random", xeta=1.0, reta=1.0, xL=None,
               xs=None) -> tuple[str, list[dict]]:
    if log is None:
        def log(msg):
            print(msg, flush=True)
    k = (n - 2) // 2
    nf = default_nf(k) if nf is None else int(nf)
    if nf_check is None:
        nf_check = 0 if nf == (1 << k) else nf + max(16, (1 << k) // 4)
    tag = tag or config_tag(n, L, s, r, R, lr, nf, ham_set, method, explore, topk, explore_mask, xeta, reta, xL, xs)
    outdir.mkdir(parents=True, exist_ok=True)
    path = ckpt(outdir, tag)
    paths = ham_paths(n, ham_set)
    if max_h is not None:
        paths = paths[:max_h]
    done = done_pairs(path)
    jobs = []
    for i, p in enumerate(paths):
        for t in range(trial_offset, trial_offset + trials):
            if (i, t) in done:
                continue
            jobs.append(dict(n=n, L=L, s=s, r=r, R=R, lr=lr, nf=nf, inst=i, trial=t, seed=trial_seed(seed0, i, t),
                             path=str(p), ham_set=ham_set, method=method, nf_check=nf_check,
                             explore=explore, topk=topk, explore_mask=explore_mask, xeta=xeta, reta=reta,
                             xL=xL, xs=xs))
    log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {tag}: {len(jobs)} jobs ({len(done)} done)")
    t0 = time.time()
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(worker, j): j for j in jobs}
            for kk, f in enumerate(as_completed(futs), 1):
                j = futs[f]
                try:
                    rec = f.result()
                except Exception as e:  # noqa: BLE001
                    log(f"FAILED {tag} inst={j['inst']} t={j['trial']}: {e!r}")
                    continue
                with open(path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                if kk % max(1, len(jobs) // 10) == 0 or kk == len(jobs):
                    log(f"[{time.strftime('%H:%M:%S')}] {tag} {kk}/{len(jobs)} elapsed={time.time() - t0:.0f}s "
                        f"last wall={rec['wall_s']:.1f}s")
    recs = [x for x in load_records(path) if trial_offset <= int(x["trial"]) < trial_offset + trials
            and int(x["inst"]) < len(paths)]
    return tag, recs


def summarize(recs: list[dict]) -> dict:
    """Aggregate metrics incl. per-round curves (round j = best-so-far after rounds 0..j)."""
    if not recs:
        return {}
    R = max(len(x["rounds"]) for x in recs) - 1
    out = {
        "n_trials": len(recs), "n_instances": len({x["inst"] for x in recs}),
        "success": float(np.mean([x["success"] for x in recs])),
        "mean_p_gs": float(np.mean([x["p_gs"] for x in recs])),
        "median_p_gs": float(np.median([x["p_gs"] for x in recs])),
        "evals": float(np.mean([x["nfev"] for x in recs])),
        "wall_s_mean": float(np.mean([x["wall_s"] for x in recs])),
        "leakage_max": float(max(x["leakage"] for x in recs)),
        "round0_polished_hit": float(np.mean([x["rounds"][0]["polished_is_ground"] for x in recs])),
        "round0_raw_hit": float(np.mean([x["rounds"][0]["success"] for x in recs])),
    }
    succ = [x for x in recs if x["success"]]
    out["mean_p_gs_given_success"] = float(np.mean([x["p_gs"] for x in succ])) if succ else 0.0
    curve = []
    for j in range(R + 1):
        # selection restricted to rounds 0..j (common eta over those rounds as in the trial's own bsf)
        rs = [x["rounds"][min(j, len(x["rounds"]) - 1)] for x in recs]
        curve.append({
            "round": j,
            "success": float(np.mean([q["bsf_success"] for q in rs])),
            "mean_p_gs": float(np.mean([q["bsf_p_gs"] for q in rs])),
            "evals": float(np.mean([q["cum_nfev"] for q in rs])),
            "guess_hit": float(np.mean([q["next_guess_is_ground"] for q in rs])),
            "round_p_gs": float(np.mean([q["p_gs"] for q in rs])),
            "round_success": float(np.mean([q["success"] for q in rs])),
        })
    out["curve"] = curve
    per_h = {}
    for x in recs:
        per_h.setdefault(x["inst"], []).append(x)
    out["per_h"] = {int(h): {"success": float(np.mean([y["success"] for y in v])),
                             "mean_p_gs": float(np.mean([y["p_gs"] for y in v]))} for h, v in sorted(per_h.items())}
    out["worst_h_mean_p_gs"] = float(min(v["mean_p_gs"] for v in out["per_h"].values()))
    if "p_gs_check" in recs[0]:
        out["nf_check_max_abs_dp"] = float(max(x["max_abs_dp_check"] for x in recs))
        out["nf_check_success_agree"] = float(np.mean([x["success_check"] == x["success"] for x in recs]))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--s", type=int, default=10)
    p.add_argument("--r", type=int, default=200)
    p.add_argument("--R", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--nf", type=int, default=None, help="Fock levels per cavity (default 2^k + max(2^k/4, 16))")
    p.add_argument("--nf-check", type=int, default=None)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--trial-offset", type=int, default=0)
    p.add_argument("--max-h", type=int, default=None)
    p.add_argument("--ham-set", choices=("scaling", "legacy8"), default="scaling")
    p.add_argument("--method", choices=("fast", "legacy"), default="fast")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--outdir", default=str(RUN_ROOT))
    p.add_argument("--tag", default=None)
    p.add_argument("--explore", type=int, default=1, help="explore growth runs (1 = old relayout protocol)")
    p.add_argument("--topk", type=int, default=1, help="top-K code states per round added to the candidate pool")
    p.add_argument("--explore-mask", choices=("random", "zero", "perm"), default="random")
    p.add_argument("--xeta", type=float, default=1.0, help="eta-controller multiplier in explore growth")
    p.add_argument("--reta", type=float, default=1.0, help="eta-controller multiplier in exploit rounds")
    p.add_argument("--xL", type=int, default=None, help="explore growth depth (default L)")
    p.add_argument("--xs", type=int, default=None, help="explore steps per stage (default s)")
    a = p.parse_args(argv)
    tag, recs = run_config(n=a.n, L=a.L, s=a.s, r=a.r, R=a.R, lr=a.lr, nf=a.nf, trials=a.trials,
                           trial_offset=a.trial_offset, ham_set=a.ham_set, method=a.method, workers=a.workers,
                           seed0=a.seed, outdir=Path(a.outdir), nf_check=a.nf_check, max_h=a.max_h, tag=a.tag,
                           explore=a.explore, topk=a.topk, explore_mask=a.explore_mask, xeta=a.xeta, reta=a.reta,
                           xL=a.xL, xs=a.xs)
    sm = summarize(recs)
    sm.pop("per_h", None)
    print(json.dumps({"tag": tag, **sm}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

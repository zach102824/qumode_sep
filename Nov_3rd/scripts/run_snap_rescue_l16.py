#!/usr/bin/env python3
"""Train one L=16 SNAP-rescue arm, or aggregate the gate.

Does not read or write the n21 phase-2 ladder.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.hva_safe import self_check
from Nov_3rd.snap_rescue import (
    ARMS,
    RUN,
    SEEDS,
    aggregate,
    embed_check,
    eval_arm,
    eval_init,
    initial_tensors,
    load_baselines,
    load_bundles,
    load_unsat,
    log,
    seed_path,
    train_arm,
    write_payload,
)


def verify(eval_bundle) -> None:
    """Reproduce the PR #36 checkpoints and the HVA→SNAP embedding."""
    log("self-check")
    E6, _ = load_unsat(6, slice(0, 1))
    self_check(E6)
    log("self-check ok")
    info = embed_check(eval_bundle)
    log(
        f"embed fidelity {info['embed_fidelity_min']:.12f}  "
        f"power_p fidelity {info['power_p_fidelity_min']:.12f}  "
        f"seal s0 n12 |Δ| {info['seal_seed0_n12_abs_diff']:.3e}"
    )
    (RUN / "embed_check.json").write_text(__import__("json").dumps(info, indent=2))
    baselines = load_baselines()
    for seed in SEEDS:
        rec = baselines["seeds"][str(seed)]
        for arm, key, ns in (
            ("hva_long", "L16_hva", (9, 12)),
            ("walsh_long", "L16_snap", (9, 12)),
            ("warm_pad", "L4_snap", (9, 12)),
        ):
            ev = eval_arm(arm, initial_tensors(arm, seed), {n: eval_bundle[n] for n in ns})
            for n in ns:
                got = ev[str(n)]["med_eps"]
                exp = float(rec[key][f"n{n}"])
                err = abs(got - exp)
                log(f"verify {arm} s{seed} n{n} got={got:.6f} exp={exp:.6f} |Δ|={err:.3e}")
                if err > 1e-5:
                    raise SystemExit(f"checkpoint mismatch {arm} s{seed} n{n} Δ={err}")
    log("verify ok")


def train_one(arm: str, seed: int, rounds: int | None, force: bool) -> None:
    train, hold, ev = load_bundles()
    payload = train_arm(arm, seed, train, hold, rounds=rounds, force=force)
    path = seed_path(arm, seed)
    if "eval" not in payload:
        log(f"eval {arm} s{seed}")
        payload["eval"] = eval_arm(arm, payload_angles(payload), ev)
        if "init_eval" not in payload:
            log(f"init-eval {arm} s{seed}")
            payload["init_eval"] = eval_init(arm, seed, ev)
        write_payload(path, payload)
        e12 = payload["eval"]["12"]["med_eps"]
        e15 = payload["eval"]["15"]["med_eps"]
        log(f"saved {path.name} n12={e12:.4f} n15={e15:.4f} best_r={payload['best_round']}")
    else:
        log(f"kept {path.name}")


def payload_angles(payload: dict):
    from Nov_3rd.snap_rescue import lists_to_tensors

    return lists_to_tensors(payload["angles"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    RUN.mkdir(parents=True, exist_ok=True)

    if args.verify:
        _train, _hold, ev = load_bundles()
        verify(ev)
    if args.arm is not None:
        if args.seed is None or args.seed not in SEEDS:
            raise SystemExit(f"--seed must be one of {SEEDS}")
        train_one(args.arm, args.seed, args.rounds, args.force)
    if args.aggregate or (args.arm is None and not args.verify):
        summary = aggregate()
        log("OVERALL", summary.get("overall"))


if __name__ == "__main__":
    main()

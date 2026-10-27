#!/usr/bin/env python3
"""L=5 densify / Opus explore queue. Target Med→WS~14. Saves weights. Skips done."""
from __future__ import annotations
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np

RUN = Path("/workspace/qumode_sep/oct_explore/v3/openevolve/runs/n8_phase1")
sys.path.insert(0, str(RUN))
sys.path.insert(0, "/workspace/qumode_sep/oct_explore/v3/openevolve/net")
spec = importlib.util.spec_from_file_location("rp", RUN / "run_phase1.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

A = dict(arch="mp", H=128, rounds=4, steps=1400, lr=1.1e-3, bs=250,
         wd=1e-4, leakpen=0.14, tailw=1.6, amax=1.8, seed=0)
WS = 13.989
CHAMP = 38.32

def _add(q, seen, nt, phase, ov, extra):
    cid = rp.cfg_id("A", 5, nt, extra)
    if cid in seen or rp.is_done(cid):
        return
    seen.add(cid)
    q.append((cid, "A", 5, nt, phase, ov))

def opus_queue():
    """Zach+Opus priority order."""
    q, seen = [], set()
    # 1) Seed baseline A@10k seeds 1-4 with EMA
    for seed in (1, 2, 3, 4):
        _add(q, seen, 10000, "opus_seed",
             dict(seed=seed, ema_decay=0.999, steps=1400), f"opus_seed{seed}_ema")
    # also seed0 with EMA for comparison to champ
    _add(q, seen, 10000, "opus_seed",
         dict(seed=0, ema_decay=0.999, steps=1400), "opus_seed0_ema")

    # 2) n=6 warm-start: no compatible tied n6 L5 checkpoint — skipped in main with note

    # 3) L=3→L=5 curriculum (two-stage handled in run_curriculum)
    # placeholder id so we can track; actual run via special phase
    cid = "A_L5_n10000_opus_curr_L3toL5"
    if not rp.is_done(cid) and cid not in seen:
        seen.add(cid)
        q.append((cid, "A", 5, 10000, "opus_curriculum",
                  dict(curr_L3_steps=800, curr_L5_steps=1000, curr_L5_lr=6e-4, ema_decay=0.999)))

    # 4) Leakpen ramp 0.4→0.1 on A@10k and A@5k
    _add(q, seen, 10000, "opus_leakramp",
         dict(leakpen_start=0.4, leakpen_end=0.1, leakpen_frac=0.30, leakpen=0.1,
              steps=1400, ema_decay=0.999), "opus_leakramp_10k")
    _add(q, seen, 5000, "opus_leakramp",
         dict(leakpen_start=0.4, leakpen_end=0.1, leakpen_frac=0.30, leakpen=0.1,
              steps=1400, ema_decay=0.999), "opus_leakramp_5k")

    # 5) Untied init — attempted if mapper succeeds; else skipped in main

    # 6) Densify: cosine/longer, H=192 r=4, ntrain≥10k
    _add(q, seen, 10000, "opus_dense",
         dict(steps=2000, lr=1.5e-3, lr_schedule="cosine", lr_min=1e-5, grad_clip=1.0,
              ema_decay=0.999), "opus_cosine_2k")
    _add(q, seen, 10000, "opus_dense",
         dict(steps=2600, lr=1.1e-3, lr_schedule="cosine", lr_min=1e-5, grad_clip=1.0,
              ema_decay=0.999), "opus_cosine_2p6k")
    _add(q, seen, 10000, "opus_dense",
         dict(H=192, rounds=4, steps=2000, lr=1.1e-3, lr_schedule="cosine", lr_min=1e-5,
              grad_clip=1.0, ema_decay=0.999), "opus_H192_cosine")
    _add(q, seen, 10000, "opus_dense",
         dict(H=192, rounds=4, steps=2600, lr=8e-4, ema_decay=0.999), "opus_H192_s2600")
    # longer push on winner recipe
    for steps, lr in ((2000, 1.1e-3), (2600, 1.1e-3), (3200, 8e-4), (4000, 6e-4)):
        extra = f"opus_s{steps}_lr{str(lr).replace('-','m').replace('.','p')}"
        _add(q, seen, 10000, "opus_dense", dict(steps=steps, lr=lr, ema_decay=0.999), extra)

    return q

def run_curriculum(cid, ov):
    """Train L=3 A 800 steps, then L=5 1000 steps lr=6e-4 from those weights."""
    d, rj, wj, lj = rp.result_paths(cid)
    if rp.is_done(cid):
        return json.loads(rj.read_text())
    d.mkdir(parents=True, exist_ok=True)
    def log(*a):
        line = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
        print(line, flush=True)
        with open(lj, "a") as f:
            f.write(line + "\n")
    import tiecost_lib as TL
    h3 = dict(A)
    h3.update(ntrain=10000, train_ns=(8,), steps=int(ov.get("curr_L3_steps", 800)),
              ema_decay=float(ov.get("ema_decay", 0.999)))
    w3 = d / "stage_L3_weights.npz"
    log(f"CURRICULUM stage1 L=3 steps={h3['steps']}")
    meta = {"id": cid, "phase": "opus_curriculum", "anchor": "A", "L": 5, "status": "running",
            "hypers": dict(ov), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    rj.write_text(json.dumps(meta, indent=2, default=str))
    r3 = TL.train_one(8, 3, h3, screen=False, log_fn=log, weights_path=str(w3))
    h5 = dict(A)
    h5.update(ntrain=10000, train_ns=(8,), steps=int(ov.get("curr_L5_steps", 1000)),
              lr=float(ov.get("curr_L5_lr", 6e-4)),
              init_th_path=str(w3), ema_decay=float(ov.get("ema_decay", 0.999)))
    # L=3 and L=5 tied nets: NB_ACT=L changes param count — init_th may fail
    log(f"CURRICULUM stage2 L=5 steps={h5['steps']} lr={h5['lr']} trying leaf-copy from {w3}")
    # L changes NB_ACT → flat th size differs. Copy shape-matched pytree leaves.
    import jax
    from jax.flatten_util import ravel_pytree
    h5.pop("init_th_path", None)
    # Build L=5 init, overlay L=3 leaves where shapes match
    rng = __import__("numpy").random.default_rng(777 + int(h5.get("seed", 0)))
    # reuse TL internals via a tiny probe train_one path: manual init
    # Call train_one with a pre-saved overlay weights file
    overlay_path = d / "stage_L5_init_from_L3.npz"
    try:
        # reconstruct L3 and L5 param trees via TL by hijacking init sizes through short import of AN
        import sys as _sys
        _sys.path.insert(0, "/workspace/qumode_sep/oct_explore")
        _sys.path.insert(0, "/workspace/qumode_sep/oct_explore/v3")
        import arch_nets as AN
        import jax.numpy as jnp
        H=int(h5["H"]); R=int(h5["rounds"]); arch=str(h5["arch"]); FV=7
        def make_p(L_act, th_flat=None):
            # mirror tiecost init_net heads for given L
            NB=L_act; C=2; n=8; K=n//2; NU=1<<K; FM=32
            def W(i,o,s=1.0):
                return jnp.asarray(rng.normal(0, s/__import__("numpy").sqrt(max(i,1)), (i,o)).astype("float32"))
            p = AN.init_backbone(rng, arch, H, R, FV)
            p["G1"]=W(2*H,H); p["g1"]=jnp.zeros(H)
            p["G2_a"]=jnp.zeros((H, C*NB*2)); p["G2_g"]=jnp.zeros((H,1))
            p["Wa"]=jnp.zeros((H, NB*2)); p["Ws"]=jnp.zeros((H, NB))
            p["P0"]=W(H,FM,1.5); p["P1"]=W(H,FM,1.5)
            p["u0"]=jnp.asarray(rng.normal(0,0.5,FM).astype("float32")); p["u1"]=jnp.asarray(rng.normal(0,0.5,FM).astype("float32"))
            p["fq"]=jnp.zeros(FM); p["Wo"]=jnp.zeros((FM, NB))
            p["Mcav1"]=W(K*(H+7),H,1.0); p["bcav1"]=jnp.zeros(H)
            p["Wfock"]=jnp.zeros((H, NB*NU)); p["Wbit"]=jnp.asarray(W(H+7,FM,1.0)); p["bbit"]=jnp.zeros(FM)
            p["Wprod"]=jnp.zeros((9,FM,NB)); p["Wphi_a"]=jnp.zeros((NU*FM, NB*2)); p["Wc"]=jnp.zeros((H,1))
            r0 = rng.uniform(0.15,0.35,(C,NB)); ph = rng.uniform(0,2*__import__("numpy").pi,(C,NB))
            A0 = __import__("numpy").stack([r0*__import__("numpy").cos(ph), r0*__import__("numpy").sin(ph)], -1).astype("float32")
            TH0 = __import__("numpy").zeros((C,NB,NU), "float32")
            import json as _json
            PREPS=_json.load(open("/workspace/qumode_sep/oct_explore/results/oct23_exp/preps.json"))
            pr1 = PREPS[f"sd_K{K}_k1"]
            A0[:,0,:] = __import__("numpy").array(pr1["alphas"][0])[None,:]
            TH0[:,0,:] = __import__("numpy").array(pr1["thetas"][0])[None,:]
            p["A0"]=jnp.asarray(A0); p["TH0"]=jnp.asarray(TH0); p["g0"]=jnp.asarray(__import__("numpy").array([3.0],"float32"))
            th, unr = ravel_pytree(p)
            if th_flat is not None:
                p = unr(jnp.asarray(th_flat))
            return p, unr
        th3 = __import__("numpy").load(w3)["th"]
        p3, _ = make_p(3, th3)
        p5, unr5 = make_p(5, None)
        copied=0; skipped=0
        for k in p5:
            if k in p3 and __import__("numpy").asarray(p3[k]).shape == __import__("numpy").asarray(p5[k]).shape:
                p5[k] = p3[k]; copied += 1
            else:
                skipped += 1
        th5, _ = ravel_pytree(p5)
        __import__("numpy").savez_compressed(overlay_path, th=__import__("numpy").asarray(th5))
        log(f"leaf-copy done copied={copied} skipped={skipped} -> {overlay_path}")
        h5["init_th_path"] = str(overlay_path)
        r5 = TL.train_one(8, 5, h5, screen=False, log_fn=log, weights_path=str(wj))
        r5["curriculum_init"] = f"leaf_copy_{copied}"
    except Exception as e:
        log(f"CURRICULUM leaf-copy failed ({e}); cold L=5")
        h5.pop("init_th_path", None)
        r5 = TL.train_one(8, 5, h5, screen=False, log_fn=log, weights_path=str(wj))
        r5["curriculum_init"] = "cold_fallback"
    r5["stage_L3"] = {k: r3[k] for k in ("test", "val", "best_step", "sec") if k in r3}
    row = {
        "id": cid, "phase": "opus_curriculum", "anchor": "A", "L": 5, "status": "done",
        "hypers": r5.get("hypers", h5), "val": r5["val"], "test": r5["test"],
        "test_ema": r5.get("test_ema"), "val_ema": r5.get("val_ema"),
        "test_final": r5.get("test_final"),
        "cost_tied": r5.get("cost_tied"), "tie_ok": r5.get("tie_ok"),
        "mean_leak_test": r5.get("mean_leak_test"), "sec": r5["sec"] + r3["sec"],
        "best_step": r5["best_step"], "weights_path": r5.get("weights_path", str(wj)),
        "curriculum_init": r5.get("curriculum_init"), "stage_L3": r5.get("stage_L3"),
        "started": meta["started"], "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "promote": bool(r5["val"]["median_inv_p"] <= 64),
        "hot": bool(r5["val"]["median_inv_p"] <= 40),
        "beats_floor": bool(r5["test"]["median_inv_p"] < 128),
        "beats_ws": bool(r5["test"]["median_inv_p"] < WS),
    }
    rj.write_text(json.dumps(row, indent=2, default=str))
    rp.append_jsonl(row)
    rp.write_status()
    log(f"CURRICULUM DONE test={r5['test']['median_inv_p']:.2f}")
    return row

def try_untied_init():
    """Best-effort: note Part1 theta exists but shape differs — skip with note unless sizes match."""
    path = Path("/workspace/qumode_sep/oct_explore/results/oct23_exp/gnn5_n8_pure_cw_theta.npz")
    note = RUN / "UNTIED_INIT_NOTE.md"
    z = np.load(path)
    th = z["theta"]
    # our A L=5 size
    import tiecost_lib as TL
    # don't full init — read from existing champ weights size
    champ = RUN / "runs/A_L5_n10000/weights.npz"
    ours = int(np.load(champ)["th"].shape[0]) if champ.exists() else None
    msg = (f"# Untied→tied init\n\nPart1 `{path.name}` theta size={th.size}. "
           f"Tied A L=5 size={ours}.\n"
           f"**Skip:** shapes differ; no safe layer-average map without shared pytree. "
           f"Part1 Med was ~12 (untied). Revisit with explicit key-wise mapper later.\n")
    note.write_text(msg)
    print(msg, flush=True)
    return None

def main():
    rp.RUN.mkdir(parents=True, exist_ok=True)
    # notes
    (RUN / "N6_WARMSTART_NOTE.md").write_text(
        "# n=6→n=8 warm-start\n\nNo compatible **tied** n=6 L=5 checkpoint "
        "(OE did not save weights; Part1 `gnn5_n6_pure_cw_theta.npz` is untied/different layout).\n"
        "**Skipped** per Zach guidance. Optionally train tied n=6 later then fine-tune.\n"
    )
    try_untied_init()
    q = opus_queue()
    print(f"OPUS+L5 queue={len(q)} (champ={CHAMP}, target WS~{WS})", flush=True)
    (RUN / "DENSE_START").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f" opus_l5 n={len(q)}\n")
    rp.write_status()
    st = (RUN / "STATUS.md").read_text()
    if "OPUS explore" not in st:
        (RUN / "STATUS.md").write_text(st.rstrip() + f"\n\n## OPUS explore\n- Queue {len(q)}: seeds+EMA → curriculum → leakramp → cosine/H192 densify.\n- n6 warm-start SKIPPED; untied init SKIPPED (size mismatch). Target Med→~14.\n")

    for cid, tag, L, nt, phase, ov in q:
        if phase == "opus_curriculum":
            row = run_curriculum(cid, ov)
        else:
            row = rp.run_one(cid, tag, L, nt, phase=phase, overrides=ov)
        if row and row.get("status") == "done":
            med = row["test"]["median_inv_p"]
            ema = (row.get("test_ema") or {}).get("median_inv_p")
            flag = ""
            if med <= 25: flag = " Med<=25"
            elif med < CHAMP: flag = " NEW_BEST"
            if ema is not None and ema < med: flag += f" ema={ema:.2f}"
            print(f"RESULT {cid} test={med:.2f}{flag}", flush=True)
            if med <= 25 or med < CHAMP:
                with open(RUN / "MILESTONES.log", "a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} med={med:.3f} ema={ema} {cid}\n")

    # after opus: light leftover densify from old list (ntrain near 10k only)
    for nt in (7000, 8000, 9000, 9500):
        cid = rp.cfg_id("A", 5, nt, f"L5_n{nt}")
        if not rp.is_done(cid):
            rp.run_one(cid, "A", 5, nt, phase="L5_ntrain", overrides={})

    rp.write_status()
    (RUN / "DENSE_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + " opus_l5\n")
    print("OPUS+L5 COMPLETE", flush=True)

if __name__ == "__main__":
    main()

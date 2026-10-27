#!/usr/bin/env python3
"""L=5 densify / Opus explore + round-2 priority queue. Target Med→WS~14. Saves weights. Skips done."""
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
CHAMP_W = RUN / "runs/A_L5_n10000/weights.npz"
BASIN_VAL = 80.0  # promote screen hits with val Med < this
SCREEN_SEEDS = list(range(0, 20))  # 0..19; TODO was 10..19 if only 0..9 first — full 20 now
REVLEAK_SEEDS = list(range(0, 10))

def _add(q, seen, nt, phase, ov, extra):
    cid = rp.cfg_id("A", 5, nt, extra)
    if cid in seen or rp.is_done(cid):
        return
    seen.add(cid)
    q.append((cid, "A", 5, nt, phase, ov))

def opus_queue():
    """Zach+Opus round-1 order. H192 / off-recipe lr grid cancelled (Opus r2); keep s2600 control."""
    q, seen = [], set()
    # 1) Seed baseline A@10k seeds 1-4 with EMA
    for seed in (1, 2, 3, 4):
        _add(q, seen, 10000, "opus_seed",
             dict(seed=seed, ema_decay=0.999, steps=1400), f"opus_seed{seed}_ema")
    _add(q, seen, 10000, "opus_seed",
         dict(seed=0, ema_decay=0.999, steps=1400), "opus_seed0_ema")

    # 3) L=3→L=5 curriculum
    cid = "A_L5_n10000_opus_curr_L3toL5"
    if not rp.is_done(cid) and cid not in seen:
        seen.add(cid)
        q.append((cid, "A", 5, 10000, "opus_curriculum",
                  dict(curr_L3_steps=800, curr_L5_steps=1000, curr_L5_lr=6e-4, ema_decay=0.999)))

    # 4) Leakpen ramp 0.4→0.1
    _add(q, seen, 10000, "opus_leakramp",
         dict(leakpen_start=0.4, leakpen_end=0.1, leakpen_frac=0.30, leakpen=0.1,
              steps=1400, ema_decay=0.999), "opus_leakramp_10k")
    _add(q, seen, 5000, "opus_leakramp",
         dict(leakpen_start=0.4, leakpen_end=0.1, leakpen_frac=0.30, leakpen=0.1,
              steps=1400, ema_decay=0.999), "opus_leakramp_5k")

    # 6) Densify: keep cosine jobs + s2600 constant-lr control; CANCEL H192 + lr grid
    _add(q, seen, 10000, "opus_dense",
         dict(steps=2000, lr=1.5e-3, lr_schedule="cosine", lr_min=1e-5, grad_clip=1.0,
              ema_decay=0.999), "opus_cosine_2k")
    _add(q, seen, 10000, "opus_dense",
         dict(steps=2600, lr=1.1e-3, lr_schedule="cosine", lr_min=1e-5, grad_clip=1.0,
              ema_decay=0.999), "opus_cosine_2p6k")
    # control: replay champ path then continue at constant lr
    _add(q, seen, 10000, "opus_dense",
         dict(steps=2600, lr=1.1e-3, ema_decay=0.999), "opus_s2600_lr0p0011")
    # CANCELLED (Opus r2): opus_H192_cosine, opus_H192_s2600,
    #   opus_s2000_lr0p0011, opus_s3200_lr0p0008, opus_s4000_lr0p0006
    return q

def anneal_overrides(init_path, seed):
    return dict(
        init_th_path=str(init_path),
        lr_schedule="cosine", lr=3e-4, lr_min=1e-5, steps=2000,
        ema_decay=0.995, seed=int(seed),
        # Anchor A base retained via hypers_of; explicit keepers:
        H=128, rounds=4, leakpen=0.14, tailw=1.6, wd=1e-4, bs=250, amax=1.8,
    )

def opus_r2_static_queue():
    """Round-2 priority: anneal champ → basin screen → reverse leak ramp."""
    q, seen = [], set()
    # 1) Anneal champion seeds 0 and 1
    if not CHAMP_W.exists():
        print(f"WARN champ weights missing: {CHAMP_W}", flush=True)
    else:
        for seed in (0, 1):
            _add(q, seen, 10000, "opus_r2_anneal",
                 anneal_overrides(CHAMP_W, seed), f"r2_anneal_s{seed}")

    # 2) Basin screen seeds 0..19 @700 steps recipe A
    for seed in SCREEN_SEEDS:
        _add(q, seen, 10000, "opus_r2_screen",
             dict(seed=seed, steps=700, ntrain=10000, ema_decay=0.999,
                  H=128, rounds=4, lr=1.1e-3),
             f"r2_screen_s{seed}")

    # 3) Reverse leak ramp: leakpen 0→0.14 over first 30%, 10 seeds @700
    for seed in REVLEAK_SEEDS:
        _add(q, seen, 10000, "opus_r2_revleak",
             dict(seed=seed, steps=700, ntrain=10000, ema_decay=0.999,
                  H=128, rounds=4, lr=1.1e-3,
                  leakpen_start=0.0, leakpen_end=0.14, leakpen_frac=0.30, leakpen=0.14),
             f"r2_revleak_s{seed}")
    return q

def queue_anneal_from_hit(hit_cid, hit_row, seed, seen_ids):
    """Schedule anneal from a promoted screen/revleak hit. Returns job tuple or None."""
    wpath = hit_row.get("weights_path") or str(RUN / "runs" / hit_cid / "weights.npz")
    if not Path(wpath).exists():
        print(f"WARN no weights for hit {hit_cid}; skip anneal", flush=True)
        return None
    extra = f"r2_anneal_from_{hit_cid.split('_', 3)[-1]}" if False else f"r2_anneal_hit_s{seed}"
    # unique per source hit
    cid = rp.cfg_id("A", 5, 10000, f"r2_anneal_hit_{hit_cid.replace('A_L5_n10000_', '')}")
    if cid in seen_ids or rp.is_done(cid):
        return None
    seen_ids.add(cid)
    ov = anneal_overrides(wpath, seed)
    return (cid, "A", 5, 10000, "opus_r2_anneal", ov)

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
    log(f"CURRICULUM stage2 L=5 steps={h5['steps']} lr={h5['lr']} trying leaf-copy from {w3}")
    import jax
    from jax.flatten_util import ravel_pytree
    h5.pop("init_th_path", None)
    overlay_path = d / "stage_L5_init_from_L3.npz"
    try:
        import sys as _sys
        _sys.path.insert(0, "/workspace/qumode_sep/oct_explore")
        _sys.path.insert(0, "/workspace/qumode_sep/oct_explore/v3")
        import arch_nets as AN
        import jax.numpy as jnp
        H=int(h5["H"]); R=int(h5["rounds"]); arch=str(h5["arch"]); FV=7
        def make_p(L_act, th_flat=None):
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
        rng = __import__("numpy").random.default_rng(777 + int(h5.get("seed", 0)))
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
    path = Path("/workspace/qumode_sep/oct_explore/results/oct23_exp/gnn5_n8_pure_cw_theta.npz")
    note = RUN / "UNTIED_INIT_NOTE.md"
    z = np.load(path)
    th = z["theta"]
    champ = RUN / "runs/A_L5_n10000/weights.npz"
    ours = int(np.load(champ)["th"].shape[0]) if champ.exists() else None
    msg = (f"# Untied→tied init\n\nPart1 `{path.name}` theta size={th.size}. "
           f"Tied A L=5 size={ours}.\n"
           f"**Skip:** shapes differ; no safe layer-average map without shared pytree. "
           f"Part1 Med was ~12 (untied). Revisit with explicit key-wise mapper later.\n")
    note.write_text(msg)
    print(msg, flush=True)
    return None

def write_cancel_note():
    (RUN / "CANCEL_OPUS_DENSE_r2.md").write_text(
        "# Cancelled after Opus round-2\n\n"
        "Not queued anymore (single-coin-flip, ~20% hit rate):\n"
        "- `opus_H192_cosine`, `opus_H192_s2600`\n"
        "- `opus_s2000_lr0p0011`, `opus_s3200_lr0p0008`, `opus_s4000_lr0p0006`\n\n"
        "Kept: `opus_cosine_2p6k`, `opus_s2600_lr0p0011` (constant-lr control).\n"
        "Round-2 priority runs next: anneal champ → screen 0..19 → revleak 0..9.\n"
    )

def write_diag_note():
    (RUN / "DIAG_NOTE_r2.md").write_text(
        "# Diagnostics to run later (not blocking densify)\n\n"
        "From Opus round-2 experiment 3 — do **not** block the train queue on these.\n\n"
        "## 1. Existing-weights forward-pass\n"
        "- Compare `g_tied`, `w_tied`, per-layer |α| vs `amax`=1.8 for champion vs "
        "seeds 2/3/4 and collapsed runs (leak Med 300–600).\n"
        "- Plateau signature: cost effectively switched off.\n"
        "- Trigger for later `g0` sweep: champ gamma far from 3.0 while plateau stays near 3.0 or →0.\n\n"
        "## 2. Untied checkpoint checks\n"
        "- Source: `results/oct23_exp/gnn5_n8_pure_cw_theta.npz`\n"
        "- Per-layer gamma variation; replace each layer gamma/`w` with layer mean, eval Med (no train).\n"
        "- If projected Med ≲60 or gamma nearly flat → worth tied homotopy. If ≳150 or strong ramp → drop.\n\n"
        "## 3. Per-instance tied oracle\n"
        "- ~30 test formulas; optimise L=5 circuit (A, TH, g, w; ~200 params) with ~16 restarts.\n"
        "- If oracle Med ≲8 → gap is optimisation (push multi-start/anneal).\n"
        "- If oracle Med ~20–30 → 14 unreachable at tied L=5 → move to L=7/9 with screen protocol.\n\n"
        "## 4. Larger test set\n"
        "- Build ~1000-formula test with WalkSATlm reference before claiming Med ≲30 "
        "(current test n=100 has ~±13% Med noise).\n"
    )

def _result_flag(row):
    med = row["test"]["median_inv_p"]
    ema = (row.get("test_ema") or {}).get("median_inv_p")
    flag = ""
    if med <= 25: flag = " Med<=25"
    elif med < CHAMP: flag = " NEW_BEST"
    if ema is not None and ema < med: flag += f" ema={ema:.2f}"
    return med, flag

def run_job(cid, tag, L, nt, phase, ov):
    if phase == "opus_curriculum":
        return run_curriculum(cid, ov)
    return rp.run_one(cid, tag, L, nt, phase=phase, overrides=ov)

def run_queue(q, label, promote_anneal=False, seen_anneal=None):
    """Run jobs; if promote_anneal, queue anneals for screen/revleak hits (val Med < BASIN_VAL)."""
    if seen_anneal is None:
        seen_anneal = set()
    hits = []
    pending = list(q)
    i = 0
    while i < len(pending):
        cid, tag, L, nt, phase, ov = pending[i]
        i += 1
        row = run_job(cid, tag, L, nt, phase, ov)
        if not row or row.get("status") != "done":
            continue
        med, flag = _result_flag(row)
        val_med = row["val"]["median_inv_p"]
        print(f"RESULT {cid} test={med:.2f} val={val_med:.2f}{flag}", flush=True)
        if med <= 25 or med < CHAMP:
            with open(RUN / "MILESTONES.log", "a") as f:
                ema = (row.get("test_ema") or {}).get("median_inv_p")
                f.write(f"{time.strftime('%H:%M:%S')} med={med:.3f} ema={ema} {cid}\n")
        if promote_anneal and phase in ("opus_r2_screen", "opus_r2_revleak"):
            if val_med < BASIN_VAL:
                seed = int(ov.get("seed", row.get("hypers", {}).get("seed", 0)))
                job = queue_anneal_from_hit(cid, row, seed, seen_anneal)
                if job:
                    pending.append(job)
                    hits.append((cid, val_med, job[0]))
                    print(f"PROMOTE {cid} val={val_med:.2f} -> queue {job[0]}", flush=True)
                    with open(RUN / "R2_HITS.log", "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} PROMOTE {cid} "
                                f"val={val_med:.3f} test={med:.3f} -> {job[0]}\n")
            else:
                print(f"NO_PROMOTE {cid} val={val_med:.2f} (need <{BASIN_VAL})", flush=True)
    return hits

def main():
    rp.RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "N6_WARMSTART_NOTE.md").write_text(
        "# n=6→n=8 warm-start\n\nNo compatible **tied** n=6 L=5 checkpoint "
        "(OE did not save weights; Part1 `gnn5_n6_pure_cw_theta.npz` is untied/different layout).\n"
        "**Skipped** per Zach guidance. Optionally train tied n=6 later then fine-tune.\n"
    )
    try_untied_init()
    write_cancel_note()
    write_diag_note()

    q = opus_queue()
    q_r2 = opus_r2_static_queue()
    print(f"OPUS+L5 queue={len(q)} r2_queue={len(q_r2)} (champ={CHAMP}, target WS~{WS})", flush=True)
    print(f"r2 ids: {[j[0] for j in q_r2[:6]]} ... ({len(q_r2)} total)", flush=True)
    (RUN / "DENSE_START").write_text(
        time.strftime("%Y-%m-%d %H:%M:%S") + f" opus_l5+r2 n={len(q)}+{len(q_r2)}\n"
    )
    rp.write_status()
    st = (RUN / "STATUS.md").read_text()
    if "OPUS r2" not in st:
        (RUN / "STATUS.md").write_text(
            st.rstrip()
            + f"\n\n## OPUS r2\n- After remaining opus_dense: anneal champ (s0/s1) → "
              f"basin screen seeds 0..19 @700 (promote val<{BASIN_VAL} → anneal) → "
              f"revleak 0→0.14 @700 ×10 seeds.\n"
              f"- Cancelled H192 / lr-grid densify. Diag note: `DIAG_NOTE_r2.md`.\n"
        )

    # Round-1 remaining (skips done)
    run_queue(q, "opus_r1", promote_anneal=False)

    # Round-2 priority
    print(f"=== START OPUS R2 queue={len(q_r2)} ===", flush=True)
    hits = run_queue(q_r2, "opus_r2", promote_anneal=True)
    print(f"=== OPUS R2 screen/revleak promotes={len(hits)} ===", flush=True)
    for src, vm, acid in hits:
        print(f"  hit {src} val={vm:.2f} annealed_as {acid}", flush=True)

    # leftover ntrain densify (low priority)
    for nt in (7000, 8000, 9000, 9500):
        cid = rp.cfg_id("A", 5, nt, f"L5_n{nt}")
        if not rp.is_done(cid):
            rp.run_one(cid, "A", 5, nt, phase="L5_ntrain", overrides={})

    rp.write_status()
    (RUN / "DENSE_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + " opus_l5+r2\n")
    print("OPUS+L5+R2 COMPLETE", flush=True)

if __name__ == "__main__":
    main()

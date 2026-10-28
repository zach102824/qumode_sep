#!/usr/bin/env python3
"""Per-instance BFGS expressivity ceiling for n=8 L=5 tied SNAP+D->cost (NO GNN).

Optimizes A, TH, g, w directly per formula under the locked tied-cost circuit
from tiecost_lib.py. Reports whether p(GS) can reach near 1.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

ROOT = "/workspace/qumode_sep/oct_explore"
OUT_DIR = os.path.join(ROOT, "results", "oct23_exp")
RUN_DIR = os.path.join(ROOT, "v3", "openevolve", "runs", "n8_phase1")
PREPS = json.load(open(os.path.join(OUT_DIR, "preps.json")))

N, L, C, K = 8, 5, 2, 4
NU = 1 << K  # 16
NF = 36
NF_CHK = 56
AMAX = 1.8
LEAKPEN = 0.1
G0 = 3.0
N_RESTARTS = 3
MAXITER = 250
N_HARD, N_RAND = 5, 5
SEED = 0


def _gray(i):
    return i ^ (i >> 1)


def build_grid_to_x():
    g2x = np.zeros((NU, NU), dtype=np.int64)
    for f0 in range(NU):
        for f1 in range(NU):
            bits = [(_gray(f0) >> j) & 1 for j in range(K)] + [
                (_gray(f1) >> j) & 1 for j in range(K)
            ]
            g2x[f0, f1] = sum(b << (N - 1 - v) for v, b in enumerate(bits))
    return g2x


def build_sim(nf, m_clauses, nm_scale, amax=AMAX):
    aa = np.diag(np.sqrt(np.arange(1, nf)), 1)
    lam, V = np.linalg.eigh(1j * (aa.T - aa))
    LAM = jnp.asarray(lam.astype(np.float32))
    VD = jnp.asarray(V.astype(np.complex64))
    NV = jnp.arange(nf, dtype=jnp.float32)
    v0 = np.zeros(nf, np.complex64)
    v0[0] = 1.0
    PSI0 = jnp.asarray(np.outer(v0, v0))

    def disp_batch(re, im):
        if amax > 0:
            r_raw = jnp.sqrt(re**2 + im**2 + 1e-20)
            sc = amax * jnp.tanh(r_raw / amax) / r_raw
            re, im = re * sc, im * sc
        r = jnp.sqrt(re**2 + im**2 + 1e-20)
        ph = jnp.arctan2(im, re)
        W = jnp.exp(1j * ph[:, None] * NV[None])[:, :, None] * VD[None]
        return jnp.einsum(
            "bik,bk,bjk->bij", W, jnp.exp(-1j * r[:, None] * LAM[None]), W.conj()
        )

    def sim_fn(A_b, TH_b, g_tied, w_tied, Cn_rel, Succ, Viol):
        B = Cn_rel.shape[0]
        psi = jnp.broadcast_to(PSI0, (B, nf, nf))
        Vf = Viol.astype(jnp.float32)
        cw = g_tied[:, None] + w_tied
        wx = nm_scale * jnp.einsum("bm,bmx->bx", cw, Vf).reshape(B, NU, NU)
        wx_rel = wx - jnp.max(wx.reshape(B, -1), axis=1)[:, None, None]
        cost_u = jnp.exp(-1j * wx_rel)
        for l in range(L):
            ph0 = jnp.concatenate(
                [TH_b[:, 0, l], jnp.zeros((B, nf - NU), jnp.float32)], axis=1
            )
            ph1 = jnp.concatenate(
                [TH_b[:, 1, l], jnp.zeros((B, nf - NU), jnp.float32)], axis=1
            )
            psi = psi * jnp.exp(-1j * ph0)[:, :, None] * jnp.exp(-1j * ph1)[:, None, :]
            D0 = disp_batch(A_b[:, 0, l, 0], A_b[:, 0, l, 1])
            D1 = disp_batch(A_b[:, 1, l, 0], A_b[:, 1, l, 1])
            psi = jnp.einsum("bij,bjk->bik", D0, psi)
            psi = jnp.einsum("bkj,bij->bik", D1, psi)
            psi = psi.at[:, :NU, :NU].multiply(cost_u)
        pr_valid = jnp.abs(psi[:, :NU, :NU]) ** 2
        p_succ = jnp.sum(pr_valid * Succ, axis=(1, 2))
        p_leak = 1.0 - jnp.sum(pr_valid, axis=(1, 2))
        return p_succ, p_leak, wx_rel

    return sim_fn


def prepare_instance(Es_row, cls_v, cls_s, m_clauses, nm_scale, GRID_TO_X):
    """Single-instance tensors matching tiecost_lib.prepare_split."""
    E_grid = Es_row[GRID_TO_X].astype(np.float32)
    c_grid = E_grid * nm_scale
    Cn_rel = (c_grid - np.max(c_grid)).astype(np.float32)
    Succ = (E_grid == 0).astype(np.float32)
    n_gs = int(Succ.sum())

    x_grid_flat = GRID_TO_X.reshape(-1)
    slot_bits = np.array(
        [[(x_val >> (N - 1 - s)) & 1 for x_val in x_grid_flat] for s in range(N)],
        dtype=np.uint8,
    )
    Ip = np.zeros((m_clauses, N), np.float32)
    In = np.zeros((m_clauses, N), np.float32)
    orders = np.arange(N, dtype=np.int64)
    flips = np.zeros(N, dtype=np.int64)
    inv_orders = orders.copy()
    v_new = inv_orders[cls_v]
    s_new = cls_s * (1 - 2 * flips[v_new])
    for c in range(m_clauses):
        for t in range(4):
            v = int(v_new[c, t])
            s = int(s_new[c, t])
            if s > 0:
                Ip[c, v] += 1.0
            else:
                In[c, v] += 1.0
    need_val = (s_new < 0).astype(np.uint8)
    Viol = (
        (slot_bits[v_new[:, 0]] == need_val[:, 0, None])
        & (slot_bits[v_new[:, 1]] == need_val[:, 1, None])
        & (slot_bits[v_new[:, 2]] == need_val[:, 2, None])
        & (slot_bits[v_new[:, 3]] == need_val[:, 3, None])
    ).astype(np.uint8)

    return {
        "Cn_rel": jnp.asarray(Cn_rel[None]),
        "Succ": jnp.asarray(Succ[None]),
        "Viol": jnp.asarray(Viol[None]),
        "n_gs": n_gs,
    }


def init_params(rng, m_clauses, noise=0.0):
    r = rng.uniform(0.15, 0.35, (C, L))
    ph = rng.uniform(0, 2 * np.pi, (C, L))
    A0 = np.stack([r * np.cos(ph), r * np.sin(ph)], -1).astype(np.float32)
    TH0 = np.zeros((C, L, NU), np.float32)
    pr1 = PREPS[f"sd_K{K}_k1"]
    A0[:, 0, :] = np.array(pr1["alphas"][0])[None, :]
    TH0[:, 0, :] = np.array(pr1["thetas"][0])[None, :]
    if noise > 0:
        A0 = A0 + rng.normal(0, noise, A0.shape).astype(np.float32)
        TH0 = TH0 + rng.normal(0, noise, TH0.shape).astype(np.float32)
    g = np.array(G0, dtype=np.float32)
    w = np.zeros((m_clauses,), np.float32)
    return {
        "A": jnp.asarray(A0),
        "TH": jnp.asarray(TH0),
        "g": jnp.asarray(g),
        "w": jnp.asarray(w),
    }


def pick_instances(nsol, n_hard=N_HARD, n_rand=N_RAND, seed=SEED):
    order = np.argsort(nsol, kind="stable")
    hard = order[:n_hard].tolist()
    rest = [i for i in range(len(nsol)) if i not in hard]
    rng = np.random.default_rng(seed)
    rand = rng.choice(rest, size=n_rand, replace=False).tolist()
    return hard, rand, hard + rand


def run_one(idx, pack, sim_main, sim_chk, m_clauses, n_restarts=N_RESTARTS, maxiter=MAXITER):
    Cn_rel, Succ, Viol = pack["Cn_rel"], pack["Succ"], pack["Viol"]
    n_gs = pack["n_gs"]

    def loss_fn(params):
        A_b = params["A"][None]
        TH_b = params["TH"][None]
        g_tied = params["g"][None]
        w_tied = params["w"][None]
        p_succ, p_leak, _ = sim_main(A_b, TH_b, g_tied, w_tied, Cn_rel, Succ, Viol)
        return -jnp.log(p_succ[0] + 1e-12) + LEAKPEN * p_leak[0]

    val_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # warm compile
    p0 = init_params(np.random.default_rng(0), m_clauses, noise=0.0)
    _ = val_and_grad(p0)
    _ = np.array(loss_fn(p0))

    best = None
    restarts = []
    t0 = time.time()
    for r in range(n_restarts):
        rng = np.random.default_rng(SEED + 1000 * (idx + 1) + r)
        noise = 0.0 if r == 0 else 0.05
        params0 = init_params(rng, m_clauses, noise=noise)
        x0, unravel = ravel_pytree(params0)

        def fun(x):
            p = unravel(jnp.asarray(x, dtype=jnp.float32))
            v, g = val_and_grad(p)
            gx, _ = ravel_pytree(g)
            return float(v), np.asarray(gx, dtype=np.float64)

        res = minimize(
            fun,
            np.asarray(x0, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7},
        )
        p_final = unravel(jnp.asarray(res.x, dtype=jnp.float32))
        A_b = p_final["A"][None]
        TH_b = p_final["TH"][None]
        g_tied = p_final["g"][None]
        w_tied = p_final["w"][None]
        p_succ, p_leak, _ = sim_main(A_b, TH_b, g_tied, w_tied, Cn_rel, Succ, Viol)
        p_succ = float(p_succ[0])
        p_leak = float(p_leak[0])
        # optional NF_CHK confirm
        p_succ_chk, p_leak_chk, _ = sim_chk(A_b, TH_b, g_tied, w_tied, Cn_rel, Succ, Viol)
        rec = {
            "restart": r,
            "noise": noise,
            "success": bool(res.success),
            "message": str(res.message),
            "nit": int(res.nit),
            "nfev": int(res.nfev),
            "loss": float(res.fun),
            "p_succ": p_succ,
            "p_leak": p_leak,
            "inv_p": float(1.0 / max(p_succ, 1e-12)),
            "p_succ_chk": float(p_succ_chk[0]),
            "p_leak_chk": float(p_leak_chk[0]),
            "g": float(p_final["g"]),
        }
        restarts.append(rec)
        if best is None or rec["p_succ"] > best["p_succ"]:
            best = rec
        print(
            f"  idx={idx} r={r} nit={rec['nit']} p={p_succ:.4f} 1/p={rec['inv_p']:.2f} "
            f"leak={p_leak:.4f} chk_p={rec['p_succ_chk']:.4f}",
            flush=True,
        )

    wall = time.time() - t0
    return {
        "idx": int(idx),
        "n_gs": int(n_gs),
        "best_p_succ": float(best["p_succ"]),
        "best_inv_p": float(best["inv_p"]),
        "best_leak": float(best["p_leak"]),
        "best_p_succ_chk": float(best["p_succ_chk"]),
        "best_leak_chk": float(best["p_leak_chk"]),
        "best_nit": int(best["nit"]),
        "n_restarts": int(n_restarts),
        "wall_s": float(wall),
        "restarts": restarts,
    }


def main():
    t_all = time.time()
    DATA = np.load(os.path.join(OUT_DIR, f"data_n{N}.npz"))
    m_clauses = int(DATA["m"])
    nm_scale = float(N) / float(m_clauses)
    GRID_TO_X = build_grid_to_x()
    nsol = np.asarray(DATA["test_nsol"])
    hard, rand, idxs = pick_instances(nsol)
    print(
        f"n={N} L={L} NF={NF} m={m_clauses} instances={idxs} "
        f"hard={hard} nsol_hard={nsol[hard].tolist()} rand={rand} nsol_rand={nsol[rand].tolist()}",
        flush=True,
    )

    sim_main = build_sim(NF, m_clauses, nm_scale)
    sim_chk = build_sim(NF_CHK, m_clauses, nm_scale)

    results = []
    for idx in idxs:
        pack = prepare_instance(
            DATA["test_E"][idx],
            DATA["test_cls_v"][idx],
            DATA["test_cls_s"][idx],
            m_clauses,
            nm_scale,
            GRID_TO_X,
        )
        # sanity: n_gs vs test_nsol
        assert pack["n_gs"] == int(nsol[idx]), (pack["n_gs"], int(nsol[idx]), idx)
        print(f"=== instance {idx} n_gs={pack['n_gs']} ===", flush=True)
        rec = run_one(idx, pack, sim_main, sim_chk, m_clauses)
        results.append(rec)

    invs = np.array([r["best_inv_p"] for r in results], float)
    ps = np.array([r["best_p_succ"] for r in results], float)
    aggregate = {
        "median_inv_p": float(np.median(invs)),
        "B90_inv_p": float(np.quantile(invs, 0.9)),
        "mean_inv_p": float(np.mean(invs)),
        "n_p_ge_0.5": int((ps >= 0.5).sum()),
        "n_p_ge_0.9": int((ps >= 0.9).sum()),
        "n_instances": len(results),
        "WalkSAT_look_med": 13.989,
        "WalkSAT_look_b90": 29.396,
    }
    hard_invs = [r["best_inv_p"] for r in results if r["idx"] in hard]
    hard_ok = sum(1 for v in hard_invs if v <= 30)
    hard_stuck = sum(1 for v in hard_invs if v >= 35)
    if hard_ok >= 3 and hard_stuck <= 1:
        verdict = (
            "expressivity OK — most hard instances reach 1/p ≲ 20–30 under per-instance BFGS; "
            "amortization/opt is the gap"
        )
        verdict_short = "expressivity_OK"
    elif hard_stuck >= 3:
        verdict = (
            "L=5 tied circuit expressivity wall — hard instances stuck with 1/p ≳ 35–100 "
            "even with direct BFGS"
        )
        verdict_short = "expressivity_wall"
    else:
        verdict = (
            "mixed — some hard instances near WalkSAT, others still high 1/p; "
            "expressivity borderline for L=5 tied"
        )
        verdict_short = "mixed"

    out = {
        "spec": {
            "n": N,
            "L": L,
            "C": C,
            "K": K,
            "NU": NU,
            "NF": NF,
            "NF_CHK": NF_CHK,
            "amax": AMAX,
            "leakpen": LEAKPEN,
            "g0": G0,
            "n_restarts": N_RESTARTS,
            "maxiter": MAXITER,
            "lock": "vacuum | (SNAP+D -> cost)^L with ONE tied (gamma, clause_w)",
            "loss": "-log(p_succ+1e-12) + 0.1*leak",
            "optimizer": "L-BFGS-B + JAX value_and_grad",
        },
        "instance_selection": {
            "hard_idxs": hard,
            "hard_nsol": [int(nsol[i]) for i in hard],
            "rand_idxs": rand,
            "rand_nsol": [int(nsol[i]) for i in rand],
            "all_idxs": idxs,
        },
        "results": results,
        "aggregate": aggregate,
        "verdict": verdict,
        "verdict_short": verdict_short,
        "wall_s_total": float(time.time() - t_all),
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }

    json_path = os.path.join(RUN_DIR, "ceiling_bfgs_n8_L5.json")
    md_path = os.path.join(RUN_DIR, "ceiling_bfgs_n8_L5.md")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    lines = [
        "# n=8 L=5 per-instance BFGS expressivity ceiling",
        "",
        f"Locked: vacuum | (SNAP+D → cost)^L with ONE tied (γ, clause_w). NO GNN.",
        f"NF={NF} (chk {NF_CHK}), amax={AMAX}, restarts={N_RESTARTS}, maxiter={MAXITER}.",
        f"Instances: hard={hard} (nsol={ [int(nsol[i]) for i in hard] }), rand={rand}.",
        "",
        "| idx | kind | n_gs | best p_succ | best 1/p | leak | chk p | #restarts | nit | wall_s |",
        "|-----|------|------|-------------|----------|------|-------|-----------|-----|--------|",
    ]
    for r in results:
        kind = "hard" if r["idx"] in hard else "rand"
        lines.append(
            f"| {r['idx']} | {kind} | {r['n_gs']} | {r['best_p_succ']:.4f} | "
            f"{r['best_inv_p']:.2f} | {r['best_leak']:.4f} | {r['best_p_succ_chk']:.4f} | "
            f"{r['n_restarts']} | {r['best_nit']} | {r['wall_s']:.1f} |"
        )
    lines += [
        "",
        f"**Aggregate** (best 1/p over {len(results)}): "
        f"median={aggregate['median_inv_p']:.2f}, B90={aggregate['B90_inv_p']:.2f}, "
        f"p≥0.5: {aggregate['n_p_ge_0.5']}/{len(results)}, "
        f"p≥0.9: {aggregate['n_p_ge_0.9']}/{len(results)}. "
        f"WalkSAT look Med/B90 ≈ {aggregate['WalkSAT_look_med']:.2f}/{aggregate['WalkSAT_look_b90']:.2f}.",
        "",
        f"**Verdict:** {verdict}",
        "",
        f"Wall time: {out['wall_s_total']:.1f}s (Asia/Shanghai box clock).",
        f"JSON: `{json_path}`",
    ]
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n=== DONE ===", flush=True)
    print(json.dumps(aggregate, indent=2), flush=True)
    print("VERDICT:", verdict, flush=True)
    print("wrote", json_path, md_path, flush=True)
    print(f"wall_s_total={out['wall_s_total']:.1f}", flush=True)


if __name__ == "__main__":
    main()

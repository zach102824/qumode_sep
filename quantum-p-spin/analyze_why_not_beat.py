"""Analyze why per-ZZZ / free-SNAP cannot beat QAOA much at L=8 n9 Γ=0.5."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG))

from pspin import make_dataset, gs_torch, scale_p3, all_triples
from circuit_snap import make_ansatz, forward_state, fidelity, B_SHELL
from circuit_fourier import shell_phi_from_betas
from circuit_perzz import (
    build_terms, terms_on_grid, forward_state_perzz, fidelity_perzz, expand_gamma, n_triples
)

OUT = PKG / "runs" / "perzz_explore"
DEVICE = torch.device("cpu")
N, L, GAMMA = 9, 8, 0.5
T = n_triples(N)  # 84
BASE_QAOA = 0.4380


def load_pt(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def main():
    results = {}

    # --- champs ---
    qaoa = load_pt(PKG / "runs/compare/20261105_G0.5_qaoa_vs_snap/fourier/ckpt_fourier_G0.5_L8_n9_seed2.pt")
    snap = load_pt(PKG / "runs/smoke/20261105_smoke1/ckpt_G0.5_L8_n9_seed1.pt")
    warm = load_pt(OUT / "full_xj_snap_warm/ckpt_L8_seed0.pt")
    fact = load_pt(OUT / "factorth_xj_snap/ckpt_L8_seed1.pt")

    g_q = qaoa["gamma"].detach().cpu().numpy().astype(np.float64)  # (L,)
    b_q = qaoa["beta"].detach().cpu().numpy().astype(np.float64)
    phi_q = shell_phi_from_betas(torch.as_tensor(b_q)).numpy()

    g_s = snap["gamma"].detach().cpu().numpy().astype(np.float64)
    phi_s = snap["phi"].detach().cpu().numpy().astype(np.float64)

    g_w = warm["gamma_LT"].detach().cpu().numpy().astype(np.float64)  # (L,T)
    phi_w = warm["phi"].detach().cpu().numpy().astype(np.float64)

    # factorth: expand
    raw_f = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in fact["raw"].items()}
    g_f = expand_gamma("factorth_xj_snap", {k: torch.as_tensor(v) for k,v in raw_f.items()}, L, T).numpy()
    phi_f = fact["phi"].detach().cpu().numpy().astype(np.float64)

    # ========== 1. Manifold collapse: is γ_{ℓ,t} ≈ γ_ℓ ? ==========
    def layer_fit_const(gLT):
        """Per layer: fit γ_t ≈ c, report RMSE / |c| and corr with ones."""
        rows = []
        for ell in range(L):
            v = gLT[ell]
            c = float(v.mean())
            rmse = float(np.sqrt(np.mean((v - c) ** 2)))
            rel = rmse / (abs(c) + 1e-12)
            rows.append({"ell": ell, "c": c, "rmse": rmse, "rel_rmse": rel,
                         "std": float(v.std()), "min": float(v.min()), "max": float(v.max())})
        return rows

    warm_fit = layer_fit_const(g_w)
    fact_fit = layer_fit_const(g_f)

    # Compare layer-mean γ to QAOA γ
    warm_c = np.array([r["c"] for r in warm_fit])
    fact_c = np.array([r["c"] for r in fact_fit])
    # Best global scale match warm_c ≈ a * g_q
    def best_scale(a, b):
        # a ≈ s * b
        s = float(np.dot(a, b) / (np.dot(b, b) + 1e-18))
        resid = a - s * b
        return s, float(np.sqrt(np.mean(resid**2))), float(np.corrcoef(a, b)[0,1])

    sw, rmse_w, corr_w = best_scale(warm_c, g_q)
    sf, rmse_f, corr_f = best_scale(fact_c, g_q)
    ss, rmse_s, corr_s = best_scale(g_s, g_q)

    # SNAP phi vs QAOA linear
    def phi_to_linear_rmse(phi, beta_ref=None):
        # gauge subtract phi[:,0], fit b per layer: phi'[k]≈ -b*k
        pcs = np.arange(B_SHELL, dtype=np.float64)
        rmses, bs = [], []
        for ell in range(L):
            ph = phi[ell] - phi[ell, 0]
            # fit b: ph ≈ -b * pcs  => b = -sum(ph*pcs)/sum(pcs^2) for k>=1
            num = -np.dot(ph[1:], pcs[1:])
            den = np.dot(pcs[1:], pcs[1:])
            b = num / den
            pred = -b * pcs
            rmse = float(np.sqrt(np.mean((ph - pred) ** 2)))
            rmses.append(rmse); bs.append(b)
        return float(np.mean(rmses)), np.array(bs), float(np.max(rmses))

    snap_phi_rmse, snap_bs, snap_phi_max = phi_to_linear_rmse(phi_s)
    warm_phi_rmse, warm_bs, warm_phi_max = phi_to_linear_rmse(phi_w)
    fact_phi_rmse, fact_bs, fact_phi_max = phi_to_linear_rmse(phi_f)

    results["manifold"] = {
        "qaoa_gamma": g_q.tolist(),
        "qaoa_beta": b_q.tolist(),
        "snap_gamma": g_s.tolist(),
        "snap_vs_qaoa_gamma_scale": ss, "snap_vs_qaoa_gamma_rmse": rmse_s, "snap_vs_qaoa_gamma_corr": corr_s,
        "warm_layer_mean_vs_qaoa": {"scale": sw, "rmse": rmse_w, "corr": corr_w, "layer_means": warm_c.tolist()},
        "fact_layer_mean_vs_qaoa": {"scale": sf, "rmse": rmse_f, "corr": corr_f, "layer_means": fact_c.tolist()},
        "warm_per_layer_flatness": warm_fit,
        "fact_per_layer_flatness": fact_fit,
        "warm_mean_rel_rmse": float(np.mean([r["rel_rmse"] for r in warm_fit])),
        "fact_mean_rel_rmse": float(np.mean([r["rel_rmse"] for r in fact_fit])),
        "snap_phi_linear_rmse_mean": snap_phi_rmse, "snap_phi_linear_rmse_max": snap_phi_max,
        "warm_phi_linear_rmse_mean": warm_phi_rmse, "fact_phi_linear_rmse_mean": fact_phi_rmse,
        "snap_fitted_betas": snap_bs.tolist(),
        "qaoa_betas": b_q.tolist(),
    }

    # Fraction of γ variance that is layer-constant (across triples)
    def var_explained_by_const(gLT):
        total = float(np.var(gLT))
        # residual after subtracting layer means
        resid = gLT - gLT.mean(axis=1, keepdims=True)
        resid_var = float(np.var(resid))
        return 1.0 - resid_var / (total + 1e-18), total, resid_var

    ve_w, tot_w, res_w = var_explained_by_const(g_w)
    ve_f, tot_f, res_f = var_explained_by_const(g_f)
    results["manifold"]["warm_var_explained_by_layer_const"] = ve_w
    results["manifold"]["fact_var_explained_by_layer_const"] = ve_f
    results["manifold"]["warm_total_var"] = tot_w
    results["manifold"]["warm_resid_var"] = res_w

    # ========== 2. State overlaps on hold ==========
    print("building ansatz + hold data...", flush=True)
    sim = make_ansatz(N, L, DEVICE)
    hold = make_dataset(N, GAMMA, 32, seed0=1000 * N + int(round(10 * GAMMA)) + 50_000)
    # pack
    hold_pack_hc = []
    hold_pack_xj = []
    for it in hold:
        Hc = torch.as_tensor(it.Hc_diag, dtype=torch.float32, device=DEVICE)
        gs = gs_torch(it, DEVICE)
        hold_pack_hc.append((Hc, gs))
        terms = torch.as_tensor(build_terms(it.n, it.J, "xj"), dtype=torch.float32, device=DEVICE)
        tgrid = terms_on_grid(sim, terms)
        hold_pack_xj.append((tgrid, gs))

    g_q_t = torch.as_tensor(g_q, dtype=torch.float32)
    phi_q_t = torch.as_tensor(phi_q, dtype=torch.float32)
    g_s_t = torch.as_tensor(g_s, dtype=torch.float32)
    phi_s_t = torch.as_tensor(phi_s, dtype=torch.float32)
    g_w_t = torch.as_tensor(g_w, dtype=torch.float32)
    phi_w_t = torch.as_tensor(phi_w, dtype=torch.float32)
    g_f_t = torch.as_tensor(g_f, dtype=torch.float32)
    phi_f_t = torch.as_tensor(phi_f, dtype=torch.float32)

    # Also collapsed warm: replace γ_{ℓ,t} by layer mean
    g_w_coll = np.repeat(warm_c[:, None], T, axis=1)
    g_w_coll_t = torch.as_tensor(g_w_coll, dtype=torch.float32)

    overlaps = {"snap_qaoa": [], "warm_qaoa": [], "fact_qaoa": [], "warm_coll_qaoa": [],
                "warm_coll_warm": [], "F_qaoa": [], "F_snap": [], "F_warm": [], "F_fact": [],
                "F_warm_coll": []}

    print("computing overlaps (32 hold)...", flush=True)
    with torch.no_grad():
        for i in range(32):
            Hc, gs = hold_pack_hc[i]
            tgrid, _ = hold_pack_xj[i]
            psi_q = forward_state(sim, Hc, g_q_t, phi_q_t)
            psi_s = forward_state(sim, Hc, g_s_t, phi_s_t)
            psi_w = forward_state_perzz(sim, tgrid, g_w_t, phi_w_t)
            psi_f = forward_state_perzz(sim, tgrid, g_f_t, phi_f_t)
            psi_wc = forward_state_perzz(sim, tgrid, g_w_coll_t, phi_w_t)

            def ov2(a, b):
                return float(torch.abs(torch.vdot(a, b)) ** 2)

            overlaps["snap_qaoa"].append(ov2(psi_s, psi_q))
            overlaps["warm_qaoa"].append(ov2(psi_w, psi_q))
            overlaps["fact_qaoa"].append(ov2(psi_f, psi_q))
            overlaps["warm_coll_qaoa"].append(ov2(psi_wc, psi_q))
            overlaps["warm_coll_warm"].append(ov2(psi_wc, psi_w))

            overlaps["F_qaoa"].append(float(torch.abs(torch.vdot(gs, psi_q)) ** 2))
            overlaps["F_snap"].append(float(torch.abs(torch.vdot(gs, psi_s)) ** 2))
            overlaps["F_warm"].append(float(torch.abs(torch.vdot(gs, psi_w)) ** 2))
            overlaps["F_fact"].append(float(torch.abs(torch.vdot(gs, psi_f)) ** 2))
            overlaps["F_warm_coll"].append(float(torch.abs(torch.vdot(gs, psi_wc)) ** 2))

    def summ(xs):
        a = np.asarray(xs, dtype=np.float64)
        return {"mean": float(a.mean()), "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}

    results["overlaps"] = {k: summ(v) for k, v in overlaps.items()}
    # delta F from collapsing warm γ to const
    dF = np.asarray(overlaps["F_warm"]) - np.asarray(overlaps["F_warm_coll"])
    results["overlaps"]["deltaF_warm_minus_collapsed"] = summ(dF.tolist())

    # ========== 3. Overfit table from saved results ==========
    overfit_rows = []
    # QAOA / SNAP from compare
    fo = json.load(open(PKG / "runs/compare/20261105_G0.5_qaoa_vs_snap/fourier/train_results.json"))
    for r in fo:
        if r["L"]==8 and r["mode"]=="n9" and r["seed"]==2:
            overfit_rows.append({
                "recipe": "QAOA", "n_params": 16,
                "hold": r["best_hold_score"],
                "train_med": r["train"]["9"]["median_F"],
                "gap": r["train"]["9"]["median_F"] - r["best_hold_score"],
            })
    sn = json.load(open(PKG / "runs/compare/20261105_G0.5_qaoa_vs_snap/snap/train_results.json"))
    for r in sn:
        if r["L"]==8 and r["mode"]=="n9" and r["seed"]==1:
            overfit_rows.append({
                "recipe": "SNAP_free", "n_params": 40,
                "hold": r["best_hold_score"],
                "train_med": r["train"]["9"]["median_F"],
                "gap": r["train"]["9"]["median_F"] - r["best_hold_score"],
            })
    for name in ["factor1p_xj","factor1p_xj_snap","factorth_xj_snap","lowrank_xj_snap",
                 "full_xj_snap","full_xj_snap_warm","full_xj_snap_adam","full_free_snap"]:
        p = OUT / name / "train_results.json"
        if not p.exists():
            continue
        rows = json.load(open(p))
        best = max(rows, key=lambda x: x["hold_medF"])
        overfit_rows.append({
            "recipe": name, "n_params": best["n_params"],
            "hold": best["hold_medF"],
            "train_med": best["train"]["median_F"],
            "gap": best["train"]["median_F"] - best["hold_medF"],
        })
    # wave2
    w2 = PKG / "runs/uc_explore_w2"
    for name in ["two_scale_snap_L8","poly3_snap_L8","mag_bucket_snap_L8"]:
        p = w2 / name / "train_results.json"
        if not p.exists():
            continue
        rows = json.load(open(p))
        if not rows:
            continue
        best = max(rows, key=lambda x: x.get("hold_medF", 0))
        overfit_rows.append({
            "recipe": "w2/"+name, "n_params": best.get("n_params"),
            "hold": best["hold_medF"],
            "train_med": best["train"]["median_F"],
            "gap": best["train"]["median_F"] - best["hold_medF"],
        })
    results["overfit"] = overfit_rows

    # ========== 4. Null directions at QAOA optimum ==========
    # Perturb: γ_{ℓ,t} = γ_ℓ + ε * δ_{ℓ,t}  with δ random unit (layer-zero-mean) or PCA residual of warm
    print("null-direction probes...", flush=True)
    # Use first 8 hold instances for speed
    n_probe = 8
    eps_list = [0.0, 0.05, 0.1, 0.2, 0.5]

    # Direction 1: warm residual (normalized)
    resid_w = g_w - warm_c[:, None]
    nr = np.linalg.norm(resid_w)
    dir_warm = resid_w / (nr + 1e-12)

    # Direction 2: random layer-zero-mean
    rng = np.random.default_rng(0)
    dir_rand = rng.normal(size=(L, T))
    dir_rand -= dir_rand.mean(axis=1, keepdims=True)
    dir_rand /= np.linalg.norm(dir_rand)

    # Direction 3: free form mismatch — add component orthogonal in J-weighted sense
    # δ_t proportional to random, then remove projection onto constant (xj baseline direction)
    dir_j = rng.normal(size=(L, T))
    dir_j -= dir_j.mean(axis=1, keepdims=True)
    dir_j /= np.linalg.norm(dir_j)

    def medF_gamma(gLT, phi_np, n_inst=n_probe):
        gT = torch.as_tensor(gLT, dtype=torch.float32)
        pT = torch.as_tensor(phi_np, dtype=torch.float32)
        Fs = []
        with torch.no_grad():
            for i in range(n_inst):
                tgrid, gs = hold_pack_xj[i]
                Fs.append(float(fidelity_perzz(sim, tgrid, gs, gT, pT).cpu()))
        return float(np.median(Fs))

    # Baseline: QAOA as perzz (γ_{ℓ,t}=γ_ℓ, phi=QAOA)
    g_base = np.repeat(g_q[:, None], T, axis=1)
    base_F = medF_gamma(g_base, phi_q)

    null_curves = {}
    for name, direction in [("warm_resid", dir_warm), ("rand_zm", dir_rand), ("rand_zm2", dir_j)]:
        curve = []
        for eps in eps_list:
            g_pert = g_base + eps * direction * np.linalg.norm(g_base)  # scale eps relative to ||g||
            # actually use absolute: eps * direction where direction is unit Frobenius
            g_pert = g_base + eps * direction
            curve.append({"eps": eps, "medF": medF_gamma(g_pert, phi_q)})
        null_curves[name] = curve

    # Also: move along warm residual but keep warm's free SNAP phi
    curve_wphi = []
    for eps in eps_list:
        g_pert = g_base + eps * dir_warm
        curve_wphi.append({"eps": eps, "medF": medF_gamma(g_pert, phi_w)})
    null_curves["warm_resid_with_warm_phi"] = curve_wphi

    # Finite-diff gradient of medF w.r.t. layer-constant vs residual subspace
    # Sample a few random residual directions; report |dF/deps| at eps=0
    grads = []
    for s in range(12):
        d = rng.normal(size=(L, T))
        d -= d.mean(axis=1, keepdims=True)
        d /= np.linalg.norm(d)
        f_p = medF_gamma(g_base + 0.05 * d, phi_q)
        f_m = medF_gamma(g_base - 0.05 * d, phi_q)
        grads.append(abs(f_p - f_m) / 0.1)
    # layer-constant direction (same δγ for all t)
    grads_const = []
    for s in range(8):
        dL = rng.normal(size=(L,))
        dL /= np.linalg.norm(dL)
        d = np.repeat(dL[:, None], T, axis=1)
        d /= np.linalg.norm(d)
        f_p = medF_gamma(g_base + 0.05 * d, phi_q)
        f_m = medF_gamma(g_base - 0.05 * d, phi_q)
        grads_const.append(abs(f_p - f_m) / 0.1)

    results["null"] = {
        "base_medF_8": base_F,
        "curves": null_curves,
        "grad_abs_residual_dirs": {"mean": float(np.mean(grads)), "median": float(np.median(grads)),
                                   "max": float(np.max(grads)), "all": grads},
        "grad_abs_layerconst_dirs": {"mean": float(np.mean(grads_const)), "median": float(np.median(grads_const)),
                                     "max": float(np.max(grads_const)), "all": grads_const},
        "ratio_resid_over_const_grad": float(np.mean(grads) / (np.mean(grads_const) + 1e-18)),
    }

    # ========== 5. Physics: diagonal span ==========
    print("physics span checks...", flush=True)
    # For one hold instance: columns = ZZZ_t vectors in R^{512}; H_c = -s J·ZZZ
    it0 = hold[0]
    from circuit_perzz import zzz_bitstrings
    zzz = zzz_bitstrings(N)  # (T, 512)
    # rank of ZZZ matrix
    rank_zzz = int(np.linalg.matrix_rank(zzz, tol=1e-8))
    # H_c direction
    Hc = it0.Hc_diag
    # projection of Hc onto span of ZZZ (should be exact for xj)
    # least squares: ZZZ.T @ a ≈ Hc  => a shape (T,)
    # Actually Hc = zzz.T @ (-s J) exactly
    s = scale_p3(N)
    a_true = -s * it0.J
    Hc_recon = zzz.T @ a_true
    recon_err = float(np.linalg.norm(Hc_recon - Hc) / (np.linalg.norm(Hc) + 1e-18))

    # How much of a random diagonal is in span(ZZZ)?
    rng2 = np.random.default_rng(1)
    frac_in_span = []
    for _ in range(20):
        v = rng2.normal(size=512)
        # project onto rowspan of zzz (rows are ZZZ_t)
        # P = Z^T (Z Z^T)^+ Z  where Z is (T,512)
        # use lstsq: min ||Z.T a - v||
        a, *_ = np.linalg.lstsq(zzz.T, v, rcond=None)
        proj = zzz.T @ a
        frac_in_span.append(float(np.linalg.norm(proj)**2 / (np.linalg.norm(v)**2 + 1e-18)))

    # Effective dimension: for warm γ, the diagonal phase θ_x = sum_t γ_t * (-s J_t ZZZ_t,x)
    # = sum_t (γ_t a_true_t) ZZZ_t  — still IN the same span as H_c's generators, just reweighted.
    # Compare phase vector for QAOA vs warm on instance 0, layer 0
    terms0 = build_terms(N, it0.J, "xj")  # (T, 512)
    phase_q = g_q[0] * terms0.sum(axis=0)  # γ * H_c contrib (before -i)
    phase_w = g_w[0] @ terms0
    # cosine similarity of phase vectors
    def cos(a,b):
        return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-18))
    phase_cos = [cos(g_q[ell]*terms0.sum(0), g_w[ell]@terms0) for ell in range(L)]
    phase_rel_rmse = []
    for ell in range(L):
        pq = g_q[ell]*terms0.sum(0)
        pw = g_w[ell]@terms0
        # match scale
        sc = np.dot(pw,pq)/(np.dot(pq,pq)+1e-18)
        phase_rel_rmse.append(float(np.sqrt(np.mean((pw - sc*pq)**2))/(np.std(pq)+1e-18)))

    # Hamming mixer expressivity note: SNAP only has 4 shells; popcount groups.
    # Count unique popcounts and sizes
    pop = np.array([bin(x).count("1") for x in range(512)])
    shell_sizes = {k: int((pop==k).sum()) for k in range(10)}

    results["physics"] = {
        "n": N, "dim": 512, "T_triples": T,
        "rank_ZZZ": rank_zzz,
        "Hc_recon_rel_err": recon_err,
        "random_diag_frac_in_ZZZ_span_mean": float(np.mean(frac_in_span)),
        "phase_cos_warm_vs_qaoa_per_layer": phase_cos,
        "phase_cos_mean": float(np.mean(phase_cos)),
        "phase_rel_rmse_after_scale": phase_rel_rmse,
        "phase_rel_rmse_mean": float(np.mean(phase_rel_rmse)),
        "shell_sizes": shell_sizes,
        "note": "per-ZZZ xj still produces diagonal phases in span{J_t ZZZ_t}=span{generators of H_c}; only reweights coefficients. Free (ignore J) leaves span{ZZZ} but destroys instance alignment → kill. SNAP mixer only phases Hamming shells (4 params), cannot exploit arbitrary diagonal reweighting.",
    }

    # ========== 6. Expressivity vs landscape: cold vs warm already known ==========
    results["landscape"] = {
        "full_xj_snap_cold_hold": 0.3825,
        "full_xj_snap_warm_hold": 0.4706,
        "full_free_snap_hold": 0.0984,
        "qaoa_hold": 0.4380,
        "interpretation": "704-param cold < QAOA; warm recovers +0.033 then overfits. Capacity exists on train but landscape+hold generalization blocks dramatic gains. Ignoring J kills — instance structure mandatory.",
    }

    out_json = OUT / "why_not_beat_numbers.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out_json, flush=True)
    # print key numbers
    print("=== KEY ===")
    print("warm var explained by layer-const:", ve_w)
    print("warm mean rel rmse:", results["manifold"]["warm_mean_rel_rmse"])
    print("overlaps warm-qaoa:", results["overlaps"]["warm_qaoa"])
    print("overlaps warm_coll-warm:", results["overlaps"]["warm_coll_warm"])
    print("deltaF warm-coll:", results["overlaps"]["deltaF_warm_minus_collapsed"])
    print("null grad resid/const:", results["null"]["ratio_resid_over_const_grad"])
    print("phase cos mean:", results["physics"]["phase_cos_mean"])
    return results

if __name__ == "__main__":
    main()

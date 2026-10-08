#!/usr/bin/env python3
"""Good-vs-bad layout analysis of the full encoding screen (finished Hamiltonians).

Physical slots d,e,A2,A1,A0,B2,B1,B0; record perm[i] = slot of Z(i+1); n_A = 4A2+2A1+A0.
Physical move set used for landscape features ("Fock moves"): n_A±1, n_B±1, flip d, flip e.

Stages (all cheap classical post-processing of the screen JSONLs; no quantum simulation):
  load      per-(H, class, init) p_gs / success / most-likely bitstring  (8 procs, one per H)
  features  per-(H, class) physical features of the encoded landscape   (8 procs)
  analyze   noise ceiling, trapped-trial anatomy, univariate rules, avoid filters,
            leave-one-Hamiltonian-out prefer rules, odd cases, figures, LAYOUT_PATTERNS.md data
  prereg    frozen rules applied to H8..H19 Hamiltonians (no screen data read)

  PYTHONPATH=. /workspace/venv-qumode/bin/python noiseless/analyze_layout_patterns.py --train 0-7
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from noiseless.encoding import distinct_assignments, logical_energies_from_terms, z_terms_from_npz

REPO = Path(__file__).resolve().parents[1]
SCREEN = REPO / "noiseless" / "results" / "encoding_screen" / "full"
HAMDIR = REPO / "Hamiltonians" / "four_sat"
OUT = REPO / "noiseless" / "results" / "layout_patterns"
NC, NI = 20160, 3
TRAP_P = 0.01  # trapped trial: p_gs below this (optimizer stuck on a wrong state)

PERMS = np.array(distinct_assignments(), dtype=np.int64)  # (NC, 8)
W8 = 1 << np.arange(7, -1, -1)
# physical flat index f = ((d*2+e)*8+nA)*8+nB ; physical slot bits of f
_F = np.arange(256)
PD, PE, PA, PB = _F // 128, (_F // 64) % 2, (_F // 8) % 8, _F % 8
PHYS_BITS = np.stack([PD, PE, PA >> 2 & 1, PA >> 1 & 1, PA & 1, PB >> 2 & 1, PB >> 1 & 1, PB & 1], 1)
DVAC = PD + PE + PA + PB  # Fock-move distance from vacuum


def _flat(d, e, na, nb):
    return ((d * 2 + e) * 8 + na) * 8 + nb


NEIGH = np.full((256, 7), -1, dtype=np.int64)
for f in range(256):
    d, e, a, b = PD[f], PE[f], PA[f], PB[f]
    nb = [f, _flat(1 - d, e, a, b), _flat(d, 1 - e, a, b)]
    nb += [_flat(d, e, a + s, b) for s in (-1, 1) if 0 <= a + s < 8]
    nb += [_flat(d, e, a, b + s) for s in (-1, 1) if 0 <= b + s < 8]
    NEIGH[f, : len(nb)] = nb
    NEIGH[f, len(nb):] = f  # pad with self


def logical_index():
    """(NC, 256) logical bitstring integer of each physical flat index, per layout."""
    L = PHYS_BITS[:, PERMS]  # (256, NC, 8): logical bit i = phys bit at slot perm[i]
    return np.einsum("fci,i->cf", L, W8).astype(np.int64)


def ham(h):
    terms, meta = z_terms_from_npz(HAMDIR / f"four_sat_{h:03d}.npz")
    E = logical_energies_from_terms(terms, meta["identity"])
    z = np.load(HAMDIR / f"four_sat_{h:03d}.npz")
    return E, z["clauses"].astype(int), z["polarities"].astype(int)


# ----------------------------------------------------------------------------- load
def load_one(h):
    p = np.full((NC, NI), np.nan)
    succ = np.zeros((NC, NI), bool)
    mlb = np.full((NC, NI), -1, np.int64)
    with open(SCREEN / f"four_sat_{h:03d}.jsonl") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            c, i = int(r["class_idx"]), int(r["init"]) - 1
            p[c, i] = r["p_gs"]
            succ[c, i] = r["success"]
            mlb[c, i] = int(r["most_likely_bitstring"], 2)
    return h, p, succ, mlb


# ----------------------------------------------------------------------------- features
def features_one(h):
    E, clauses, pol = ham(h)
    LI = logical_index()
    EP = E[LI]  # (NC, 256) physical energy landscape per layout
    gs_log = int(np.argmin(E))
    g = (gs_log >> np.arange(7, -1, -1)) & 1  # logical bits Z1..Z8
    ones = int(g.sum())
    minority = 1 if ones < 4 else 0  # value in the minority (ones==4: treat 0 as minority)
    gsf = np.argmax(LI == gs_log, axis=1)  # physical flat of GS per layout
    d, e, na, nb = PD[gsf], PE[gsf], PA[gsf], PB[gsf]
    df = {"h": h, "class_idx": np.arange(NC)}
    df.update(gs_d=d, gs_e=e, gs_nA=na, gs_nB=nb, gs_nsum=na + nb, gs_nmax=np.maximum(na, nb),
              gs_nmin=np.minimum(na, nb), gs_dvac=d + e + na + nb,
              gs_transmon_ones=d + e,
              gs_n_extreme=((na == 0) | (na == 7)).astype(int) + ((nb == 0) | (nb == 7)),
              gs_n_edge=np.minimum(na, 7 - na) + np.minimum(nb, 7 - nb),  # 0 = both at 0/7
              minority_on_transmons=(d == minority).astype(int) + (e == minority),
              gs_ones=np.full(NC, ones))
    # majority value -> dist of each cavity level from the majority-extreme (0 if minority=1 else 7)
    maj_level = 0 if minority == 1 else 7
    df["gs_cav_from_majority_extreme"] = np.abs(na - maj_level) + np.abs(nb - maj_level)
    # landscape: steepest descent under Fock moves
    nbE = EP[:, NEIGH]  # (NC, 256, 7)
    arg = np.argmin(nbE, axis=2)  # self at column 0 wins ties -> stops on plateaus
    nxt = NEIGH[np.arange(256)[None, :], arg]
    is_min = nxt == np.arange(256)[None, :]
    att = nxt.copy()
    for _ in range(9):
        att = np.take_along_axis(att, att, axis=1)
    rows = np.arange(NC)
    gs_att = att == gsf[:, None]
    df["vac_reaches_gs"] = gs_att[:, 0].astype(int)
    df["gs_basin_frac"] = gs_att.mean(1)
    low = DVAC <= 4
    df["gs_basin_lowphoton"] = gs_att[:, low].mean(1)  # among states within 4 moves of vacuum
    nonGS_min = is_min & ~(np.arange(256)[None, :] == gsf[:, None])
    df["n_local_min"] = nonGS_min.sum(1)
    df["n_local_min_e1"] = (nonGS_min & np.isclose(EP, 1.0)).sum(1)
    closer = DVAC[None, :] < (d + e + na + nb)[:, None]
    df["n_local_min_closer_than_gs"] = (nonGS_min & closer).sum(1)
    e1 = np.isclose(EP, 1.0)
    df["n_e1_closer_than_gs"] = (e1 & closer).sum(1)
    dv = np.where(e1, DVAC[None, :], 99)
    df["e1_min_dvac_minus_gs"] = dv.min(1) - (d + e + na + nb)
    # Fock-move distance from GS, fitness-distance correlation, roughness
    D = (np.abs(PD[None, :] - d[:, None]) + np.abs(PE[None, :] - e[:, None])
         + np.abs(PA[None, :] - na[:, None]) + np.abs(PB[None, :] - nb[:, None])).astype(float)
    Ec = EP - EP.mean(1, keepdims=True)
    Dc = D - D.mean(1, keepdims=True)
    df["fdc"] = (Ec * Dc).sum(1) / np.sqrt((Ec ** 2).sum(1) * (Dc ** 2).sum(1))
    e1D = np.where(e1, D, np.nan)
    df["e1_mean_dist_to_gs"] = np.nanmean(e1D, 1)
    df["e1_n_adjacent_gs"] = (e1 & (D == 1)).sum(1)
    hi = EP >= 1.5  # the E>=2 "bumps" on the E=1 plateau (only non-flat part of the landscape)
    df["hi_mean_dist_to_gs"] = np.nanmean(np.where(hi, D, np.nan), 1)
    df["hi_mean_dvac"] = np.nanmean(np.where(hi, DVAC[None, :].astype(float), np.nan), 1)
    df["hi_n_adjacent_gs"] = (hi & (D == 1)).sum(1)
    df["hi_mean_dvac_minus_gs"] = df["hi_mean_dvac"] - (d + e + na + nb)
    real = NEIGH != np.arange(256)[:, None]
    dE = np.abs(nbE - EP[:, :, None])
    df["roughness"] = (dE * real[None]).sum((1, 2)) / real.sum()
    df["mean_E_low_photon"] = EP[:, low].mean(1)  # mean energy of states near vacuum
    # clause structure (logical, mapped through the layout)
    J = np.zeros((8, 8))
    for cl in clauses:
        for a in cl:
            for b in cl:
                if a != b:
                    J[a, b] += 1
    deg = np.array([(clauses == i).any(1).sum() for i in range(8)])
    grp = np.select([PERMS == 0, PERMS == 1, PERMS <= 4], [0, 1, 2], 3)  # d,e,A,B per variable
    same_cav = (grp[:, :, None] == grp[:, None, :]) & (grp[:, :, None] >= 2)
    df["J_within_cavity"] = (same_cav * J[None]).sum((1, 2)) / 2
    df["J_d_e"] = J[np.argmax(PERMS == 0, 1), np.argmax(PERMS == 1, 1)]
    df["deg_on_transmons"] = deg[np.argmax(PERMS == 0, 1)] + deg[np.argmax(PERMS == 1, 1)]
    msb = (PERMS == 2) | (PERMS == 5)
    lsb = (PERMS == 4) | (PERMS == 7)
    df["deg_on_msb"] = (msb * deg[None]).sum(1)
    df["deg_on_lsb"] = (lsb * deg[None]).sum(1)
    cl_grp = grp[:, clauses]  # (NC, ncl, 4)
    spans_both = (cl_grp == 2).any(2) & (cl_grp == 3).any(2)
    df["clauses_spanning_both_cavities"] = spans_both.sum(1)
    return pd.DataFrame(df)


# ----------------------------------------------------------------------------- helpers
def rank_desc(v):
    o = np.argsort(-v, kind="stable")
    r = np.empty(len(v), int)
    r[o] = np.arange(1, len(v) + 1)
    return r


def spear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(spearmanr(a[m], b[m]).correlation)


def parse_hs(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


CACHE = OUT / "cache"


def stage_load_features(hs, procs=8):
    CACHE.mkdir(parents=True, exist_ok=True)
    with Pool(min(procs, len(hs))) as pool:
        need = [h for h in hs if not (CACHE / f"trials_H{h:02d}.npz").exists()]
        for h, p, s, m in pool.imap_unordered(load_one, need):
            np.savez_compressed(CACHE / f"trials_H{h:02d}.npz", p=p, succ=s, mlb=m)
        need = [h for h in hs if not (CACHE / f"features_H{h:02d}.parquet").exists()]
        for df in pool.imap_unordered(features_one, need):
            df.to_parquet(CACHE / f"features_H{int(df.h[0]):02d}.parquet")


def load_trials(h):
    z = np.load(CACHE / f"trials_H{h:02d}.npz")
    return z["p"], z["succ"], z["mlb"]


FEATS = None  # filled from the feature table (all non-id columns)
ID_COLS = ("h", "class_idx")


def build_table(hs):
    rows = []
    for h in hs:
        f = pd.read_parquet(CACHE / f"features_H{h:02d}.parquet")
        p, s, m = load_trials(h)
        assert not np.isnan(p).any(), f"H{h} incomplete"
        trapped = p < TRAP_P
        f["mean_p"] = p.mean(1)
        f["trap_rate"] = trapped.mean(1)
        f["fail_rate"] = (~s).mean(1)
        with np.errstate(invalid="ignore"):
            f["nontrap_p"] = np.where(trapped, np.nan, p).sum(1) / (~trapped).sum(1)
        r = rank_desc(f["mean_p"].to_numpy())
        f["rank"] = r
        f["pct"] = 1 - (r - 1) / (NC - 1)  # 1 = best
        f["good"] = (r <= NC // 10).astype(int)
        f["bad"] = (r > NC - NC // 10).astype(int)
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def noise_ceiling(hs):
    out = {}
    for h in hs:
        p, _, _ = load_trials(h)
        pair = [spear(p[:, i], p[:, j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        r1 = float(np.mean(pair))
        r3 = 3 * r1 / (1 + 2 * r1)  # Spearman-Brown: reliability of the 3-init mean
        loo = [spear(np.delete(p, i, 1).mean(1), p[:, i]) for i in range(3)]
        tr = p < TRAP_P
        out[h] = {"single_init_rank_corr": r1, "reliability_3init_mean": r3,
                  "max_explainable_spearman": float(np.sqrt(max(r3, 0))),
                  "leave_one_init_out": float(np.mean(loo)),
                  "trapped_rate_overall": float(tr.mean()),
                  "fail_rate_overall": float((~load_trials(h)[1]).mean())}
    return out


def trapped_anatomy(hs, LI):
    INV = np.empty_like(LI)
    INV[np.arange(NC)[:, None], LI] = np.arange(256)[None, :]
    recs = []
    for h in hs:
        E, _, _ = ham(h)
        gs = int(np.argmin(E))
        p, s, m = load_trials(h)
        EP = E[LI]
        gsf = INV[:, gs]
        c_idx, i_idx = np.nonzero(~s)
        b = m[c_idx, i_idx]
        f = INV[c_idx, b]
        g = gsf[c_idx]
        nbE = EP[c_idx[:, None], NEIGH[f]]
        strict_min = (nbE[:, 1:] > EP[c_idx, f][:, None] + 1e-9).all(1) | (NEIGH[f, 1:] == f[:, None]).all(1)
        # baseline: a uniformly random E=1 state of the same layout
        e1_dvac = np.array([DVAC[np.isclose(EP[c], 1.0)].mean() for c in range(NC)])
        e1_gsdist = np.array([(np.abs(PD - PD[gsf[c]]) + np.abs(PE - PE[gsf[c]]) + np.abs(PA - PA[gsf[c]])
                               + np.abs(PB - PB[gsf[c]]))[np.isclose(EP[c], 1.0)].mean() for c in range(NC)])
        ham_b = np.array([bin(int(x) ^ gs).count("1") for x in b])
        recs.append(pd.DataFrame({
            "h": h, "class_idx": c_idx, "init": i_idx + 1, "p_gs": p[c_idx, i_idx],
            "trapped": p[c_idx, i_idx] < TRAP_P, "decoy": b, "decoy_E": E[b],
            "decoy_d": PD[f], "decoy_e": PE[f], "decoy_nA": PA[f], "decoy_nB": PB[f],
            "decoy_dvac": DVAC[f], "gs_dvac": DVAC[g],
            "decoy_fock_dist_gs": np.abs(PD[f] - PD[g]) + np.abs(PE[f] - PE[g]) + np.abs(PA[f] - PA[g]) + np.abs(PB[f] - PB[g]),
            "decoy_hamming_gs": ham_b, "decoy_strict_local_min": strict_min,
            "baseline_e1_dvac": e1_dvac[c_idx], "baseline_e1_fock_dist_gs": e1_gsdist[c_idx],
            "decoy_is_allzero": b == 0, "decoy_is_allone": b == 255}))
    return pd.concat(recs, ignore_index=True)


def add_derived(T):
    ty = lambda n: np.minimum(n, 7 - n)  # cavity GS codeword type: 0=000/111 1=001/110 2=010/101 3=011/100
    T = T.copy()
    T["cavA_type"], T["cavB_type"] = ty(T.gs_nA), ty(T.gs_nB)
    T["cav_type_lo"] = np.minimum(T.cavA_type, T.cavB_type)
    T["cav_type_hi"] = np.maximum(T.cavA_type, T.cavB_type)
    T["n_cav_top2_equal"] = (T.cavA_type <= 1).astype(int) + (T.cavB_type <= 1)  # GS bit2 == bit1
    return T


# human-readable avoid filters: True = layout is REMOVED
AVOID = {
    "edge>=2 (sum of cavity codeword types >= 2)": lambda T: T.gs_n_edge >= 2,
    "a cavity has GS bit2 != bit1 (type 2/3)": lambda T: T.n_cav_top2_equal < 2,
    "both cavities have GS bit2 != bit1": lambda T: T.n_cav_top2_equal == 0,
    "both transmons carry the GS majority value": lambda T: T.minority_on_transmons == 0,
    "GS photon sum nA+nB <= 7": lambda T: T.gs_nsum <= 7,
}

SCORE_FEATS = ["gs_n_edge", "e1_mean_dist_to_gs", "gs_n_extreme", "cav_type_hi", "cav_type_lo",
               "minority_on_transmons", "gs_dvac", "e1_n_adjacent_gs", "hi_mean_dist_to_gs",
               "fdc", "gs_basin_frac", "roughness", "J_within_cavity", "deg_on_transmons",
               "deg_on_msb", "deg_on_lsb", "J_d_e", "clauses_spanning_both_cavities",
               "gs_cav_from_majority_extreme", "n_e1_closer_than_gs", "mean_E_low_photon"]


def zwithin(T, cols):
    X = T[cols].astype(float).copy()
    for h in T.h.unique():
        m = T.h == h
        sd = X[m].std().replace(0, 1)
        X[m] = (X[m] - X[m].mean()) / sd
    return X.to_numpy()


def avoid_table(T):
    rows = []
    for name, fn in AVOID.items():
        rem = fn(T).to_numpy()
        per = []
        for h in sorted(T.h.unique()):
            m = (T.h == h).to_numpy()
            g, b = T.good.to_numpy()[m] == 1, T.bad.to_numpy()[m] == 1
            per.append((rem[m][b].mean(), rem[m][g].mean(), rem[m].mean(),
                        T.mean_p.to_numpy()[m][~rem[m]].mean() if (~rem[m]).any() else np.nan,
                        T.mean_p.to_numpy()[m].mean()))
        per = np.array(per)
        rows.append({"rule": name, "bad_removed": per[:, 0].mean(), "bad_removed_min": per[:, 0].min(),
                     "good_removed": per[:, 1].mean(), "good_removed_max": per[:, 1].max(),
                     "all_removed": per[:, 2].mean(), "kept_mean_p": np.nanmean(per[:, 3]),
                     "all_mean_p": per[:, 4].mean()})
    return pd.DataFrame(rows)


def eval_scores(T, score, k=5):
    """Per-H evaluation of a score (higher = predicted better); ties broken by class_idx."""
    out = []
    for h in sorted(T.h.unique()):
        m = (T.h == h).to_numpy()
        s, p, cls = score[m], T.mean_p.to_numpy()[m], T.class_idx.to_numpy()[m]
        order = np.lexsort((cls, -s))
        top = order[:k]
        r = rank_desc(p)
        top1pct = order[: NC // 100]
        out.append({"h": int(h), "spearman": spear(s, p), "top1_p": float(p[top[0]]),
                    "top1_rank": int(r[top[0]]), f"top{k}_mean_p": float(p[top].mean()),
                    f"top{k}_best_rank": int(r[top].min()), "top1pct_mean_p": float(p[top1pct].mean()),
                    "identity_p": float(p[cls == 0][0]), "random_p": float(p.mean()), "best_p": float(p.max()),
                    "top_classes": [int(c) for c in cls[top]]})
    return pd.DataFrame(out)


def loho(T, feats=SCORE_FEATS):
    """Leave-one-Hamiltonian-out prefer rules: fit on 7 Hs, score held-out H."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    hs = sorted(T.h.unique())
    X = zwithin(T, feats)
    y = T.pct.to_numpy()
    res = {k: np.zeros(len(T)) for k in ("single_best", "rule_edge", "ridge_all", "ridge_small", "gbm_ref")}
    picked = {}
    for h in hs:
        te = (T.h == h).to_numpy()
        tr = ~te
        # single feature: best mean within-H Spearman on train, sign from train
        best, bs = None, 0
        for j, f in enumerate(feats):
            rr = np.nanmean([spear(X[(T.h == g).to_numpy(), j], y[(T.h == g).to_numpy()]) for g in hs if g != h])
            if abs(rr) > abs(bs):
                best, bs = j, rr
        res["single_best"][te] = np.sign(bs) * X[te, best]
        picked.setdefault("single_best", []).append(feats[best])
        # rule: -edge, tie-break by small ridge
        rs = Ridge(1.0).fit(X[tr][:, :6], y[tr])
        small = rs.predict(X[te][:, :6])
        res["ridge_small"][te] = small
        res["rule_edge"][te] = -T.gs_n_edge.to_numpy()[te] + 1e-3 * (small - small.min()) / (np.ptp(small) + 1e-12)
        res["ridge_all"][te] = Ridge(1.0).fit(X[tr], y[tr]).predict(X[te])
        g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0)
        res["gbm_ref"][te] = g.fit(X[tr], y[tr]).predict(X[te])
    return res, picked


def odd_cases(T):
    out = {}
    E3, c3, p3 = ham(3)
    E4, c4, p4 = ham(4)
    s3 = {tuple(sorted(zip(c, p))) for c, p in zip(c3.tolist(), p3.tolist())}
    s4 = {tuple(sorted(zip(c, p))) for c, p in zip(c4.tolist(), p4.tolist())}
    out["H3_H4"] = {"shared_clauses": len(s3 & s4), "n_clauses": [len(s3), len(s4)],
                    "same_gs": int(np.argmin(E3)) == int(np.argmin(E4)),
                    "energy_corr_256": float(np.corrcoef(E3, E4)[0, 1]),
                    "n_states_same_energy": int((np.isclose(E3, E4)).sum()),
                    "per_class_spearman": spear(T[T.h == 3].mean_p.to_numpy(), T[T.h == 4].mean_p.to_numpy())}
    for h in sorted(T.h.unique()):
        r = T[(T.h == h) & (T.class_idx == 0)].iloc[0]
        out[f"identity_H{h}"] = {k: (float(r[k]) if not isinstance(r[k], str) else r[k]) for k in
                                 ("rank", "mean_p", "gs_d", "gs_e", "gs_nA", "gs_nB", "cavA_type", "cavB_type",
                                  "gs_n_edge", "minority_on_transmons")}
    return out


def fit_frozen(T):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    X = zwithin(T, SCORE_FEATS)
    ridge = Ridge(1.0).fit(X, T.pct.to_numpy())
    gbm = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0).fit(X, T.pct.to_numpy())
    return ridge, gbm


def prereg(T, test_hs):
    ridge, gbm = fit_frozen(T)
    with Pool(8) as pool:
        dfs = pool.map(features_one, test_hs)
    out = {"created_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "created_unix": time.time(),
           "trained_on": sorted(int(h) for h in T.h.unique()),
           "note": "Frozen before the screen resumed; no screen results for these Hamiltonians were read "
                   "(H8 was 83% screened on disk at freeze time but its JSONL was not opened).",
           "avoid_rule": "remove layouts where either cavity's GS codeword has bit2 != bit1 "
                         "(GS Fock level n_A or n_B in {2,3,4,5})",
           "prefer_rule_ridge": {"features_zscored_within_H": SCORE_FEATS,
                                 "coef": [float(c) for c in ridge.coef_], "intercept": float(ridge.intercept_)},
           "tier_rule": "rank by gs_n_edge = min(nA,7-nA)+min(nB,7-nB) ascending (0 best)",
           "hamiltonians": {}}
    for df in dfs:
        h = int(df.h[0])
        df = add_derived(df)
        X = zwithin(df, SCORE_FEATS)
        s_r, s_g = ridge.predict(X), gbm.predict(X)
        cls = df.class_idx.to_numpy()
        avoid = (df.n_cav_top2_equal < 2).to_numpy()
        top_r = cls[np.lexsort((cls, -s_r))][:5]
        ok = ~avoid
        top_r_ok = cls[ok][np.lexsort((cls[ok], -s_r[ok]))][:5]
        top_g = cls[np.lexsort((cls, -s_g))][:5]
        out["hamiltonians"][f"H{h}"] = {
            "gs_bitstring": format(int(np.argmin(ham(h)[0])), "08b"),
            "avoid_fraction": float(avoid.mean()), "identity_avoided": bool(avoid[0]),
            "identity_gs_n_edge": int(df.gs_n_edge[0]),
            "ridge_top5": [int(c) for c in top_r], "ridge_top5_after_avoid": [int(c) for c in top_r_ok],
            "gbm_ref_top5": [int(c) for c in top_g],
            "n_tier0_layouts": int((df.gs_n_edge == 0).sum()),
            "top5_perms": {int(c): PERMS[c].tolist() for c in top_r}}
    return out


def figures(T, A, U):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for h in sorted(T.h.unique()):
        g = T[T.h == h].groupby("gs_n_edge").mean_p.mean()
        ax[0].plot(g.index, g.values, "o-", label=f"H{h}")
    ax[0].set_xlabel("gs_n_edge = min(nA,7-nA)+min(nB,7-nB) of the GS")
    ax[0].set_ylabel("mean p(GS) (avg over layouts)")
    ax[0].legend(ncol=2, fontsize=8)
    ax[0].set_title("GS at the Fock-ladder edges is good")
    tab = T.groupby(["cav_type_lo", "cav_type_hi"]).mean_p.mean().unstack()
    im = ax[1].imshow(tab.values, cmap="viridis", origin="lower")
    ax[1].set_xticks(range(4)); ax[1].set_yticks(range(4))
    ax[1].set_xlabel("worse cavity type (0=000/111,1=001/110,2=010/101,3=011/100)")
    ax[1].set_ylabel("better cavity type")
    for (i, j), v in np.ndenumerate(tab.values):
        if np.isfinite(v):
            ax[1].text(j, i, f"{v:.2f}", ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax[1]); ax[1].set_title("mean p(GS) by GS cavity codeword types (H0-H7)")
    fig.tight_layout(); fig.savefig(OUT / "gs_edge_effect.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    tr = A[A.trapped]
    ax[0].hist([tr.decoy_dvac, tr.baseline_e1_dvac], bins=range(0, 16), label=["trapped-trial end state", "mean over all E=1 states (same layout)"])
    ax[0].set_xlabel("Fock-move distance from vacuum (d+e+nA+nB)"); ax[0].legend(); ax[0].set_title("Trapped trials never leave the vacuum")
    nt = A[~A.trapped]
    ax[1].hist(nt.decoy_hamming_gs, bins=range(0, 9)); ax[1].set_xlabel("Hamming distance of wrong winner to GS")
    ax[1].set_title("Non-trapped failures: GS neighbour wins")
    fig.tight_layout(); fig.savefig(OUT / "failure_anatomy.png", dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="0-7")
    ap.add_argument("--prereg", default="8-19")
    a = ap.parse_args(argv)
    hs = parse_hs(a.train)
    stage_load_features(hs)
    T = add_derived(build_table(hs))
    res = {"train": hs, "trap_threshold": TRAP_P, "noise": noise_ceiling(hs)}
    A = trapped_anatomy(hs, logical_index())
    res["anatomy"] = {
        "n_failed": int(len(A)), "n_trapped": int(A.trapped.sum()),
        "trapped_by_h": A[A.trapped].groupby("h").size().to_dict(),
        "means": A.groupby("trapped")[["decoy_E", "decoy_dvac", "baseline_e1_dvac", "gs_dvac", "decoy_fock_dist_gs",
                                       "baseline_e1_fock_dist_gs", "decoy_hamming_gs", "decoy_strict_local_min",
                                       "decoy_is_allzero"]].mean().to_dict(),
        "trapped_decoy_dvac_le1": float((A[A.trapped].decoy_dvac <= 1).mean()),
        "nontrapped_hamming1": float((A[~A.trapped].decoy_hamming_gs == 1).mean())}
    feats = [f for f in SCORE_FEATS + ["gs_nmax", "gs_nmin", "gs_nsum", "gs_transmon_ones", "n_local_min"]]
    U = []
    for f in feats:
        r = {tgt: [spear(T[T.h == h][f].to_numpy(float), T[T.h == h][tgt].to_numpy(float)) for h in hs]
             for tgt in ("mean_p", "nontrap_p", "trap_rate")}
        g, b = T[T.good == 1][f].mean(), T[T.bad == 1][f].mean()
        U.append({"feature": f, "rho_mean_p_median": np.nanmedian(r["mean_p"]), "rho_mean_p_min": np.nanmin(r["mean_p"]),
                  "rho_mean_p_max": np.nanmax(r["mean_p"]), "rho_nontrap_p_median": np.nanmedian(r["nontrap_p"]),
                  "rho_trap_rate_median": np.nanmedian(r["trap_rate"]), "top10_mean": g, "bottom10_mean": b})
    U = pd.DataFrame(U).sort_values("rho_mean_p_median")
    U.to_csv(OUT / "univariate.csv", index=False)
    av = avoid_table(T)
    av.to_csv(OUT / "avoid_rules.csv", index=False)
    loho_scores, picked = loho(T)
    ev = {k: eval_scores(T, s) for k, s in loho_scores.items()}
    pd.concat([e.assign(model=k) for k, e in ev.items()]).to_csv(OUT / "loho_eval.csv", index=False)
    res["loho_summary"] = {k: e.drop(columns=["h", "top_classes"]).mean().to_dict() for k, e in ev.items()}
    res["loho_single_feature_picked"] = picked
    res["tier_table"] = T.groupby(["cav_type_lo", "cav_type_hi"]).agg(
        frac=("mean_p", "size"), mean_p=("mean_p", "mean"), good=("good", "mean"), bad=("bad", "mean"),
        trap=("trap_rate", "mean")).assign(frac=lambda x: x.frac / len(T)).reset_index().to_dict("records")
    res["odd_cases"] = odd_cases(T)
    gsloc = {}
    for h in hs:  # does the GS physical location alone explain the reproducible layout effect?
        p, _, _ = load_trials(h)
        S = T[T.h == h].assign(p12=p[:, :2].mean(1), p3=p[:, 2])
        key = ["gs_d", "gs_e", "gs_nA", "gs_nB"]
        gm12 = S.groupby(key).p12.transform("mean").to_numpy()
        gsloc[h] = {"n_locations": int(S.groupby(key).ngroups),
                    "gsloc_mean_init12_vs_init3": spear(gm12, S.p3.to_numpy()),
                    "layout_mean_init12_vs_init3": spear(S.p12.to_numpy(), S.p3.to_numpy()),
                    "gsloc_insample_vs_mean_p": spear(S.groupby(key).mean_p.transform("mean").to_numpy(),
                                                      S.mean_p.to_numpy())}
    res["gs_location_only"] = gsloc
    figures(T, A, U)
    pd.concat([T[["h", "class_idx"] + SCORE_FEATS + ["mean_p", "trap_rate", "nontrap_p", "rank"]]]).to_parquet(
        OUT / "layout_features_H0_H7.parquet")
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    if a.prereg:
        pr = prereg(T, parse_hs(a.prereg))
        (OUT / "preregistered_H8_H19.json").write_text(json.dumps(pr, indent=1))
    print(json.dumps({k: res[k] for k in ("noise", "anatomy", "loho_summary")}, indent=1, default=float))
    print(av.round(3).to_string())
    print(json.dumps(res["odd_cases"], indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

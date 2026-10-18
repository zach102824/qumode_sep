"""Architecture backbones for GNN-SNAP Phase 1 scan (16:09 CST 2026-10-18).

All arches produce the same heads used by gnn_snap.py:
  hv (B,n,H)  variable embeddings  -> free-SNAP / var_mix
  hc (B,m,H)  clause embeddings    -> clause_w
  g  (B,2H)   pooled [mean hv, mean hc] -> delta on shared base angles

Arches (formula-only features; size-free):
  mp     : message-passing GNN (baseline; mp64 = H=64 R=3, mp128 = H=128 R=5)
  attn   : attention-weighted message passing over bipartite neighbours
  tf     : transformer self-attention over clauses; vars pool from clauses
  ds     : DeepSets — each node sees only local feats + global pool
  mlp    : hand-crafted local features + MLP (no learned message passing)
"""
from __future__ import annotations
import numpy as np
import jax.numpy as jnp


def _W(rng, i, o, s=1.0):
    return jnp.asarray(rng.normal(0, s / np.sqrt(max(i, 1)), (i, o)).astype(np.float32))


def _zeros(*shape):
    return jnp.zeros(shape, jnp.float32)


def _masked_softmax(scores, mask, axis):
    """mask in {0,1}; softmax over axis, zeros on non-edges."""
    scores = jnp.where(mask > 0, scores, jnp.float32(-1e9))
    return jnp.where(mask > 0, jax_softmax(scores, axis), jnp.float32(0.0))


def jax_softmax(x, axis):
    x = x - jnp.max(x, axis=axis, keepdims=True)
    e = jnp.exp(x)
    return e / (jnp.sum(e, axis=axis, keepdims=True) + 1e-12)


# Hand-crafted feature widths for mlp
FVH, FCH = 8, 6


def handcraft(G):
    """Size-free local features from Ip/In/fv. Returns fv_h (B,n,FVH), fc_h (B,m,FCH)."""
    Ip, In, fv = G["Ip"], G["In"], G["fv"]
    B, m, n = Ip.shape
    deg = G["deg"]  # (B,n)
    # var: [1, pos_n, neg_n, pos/(pos+neg), mean nbr-clause pos-frac, mean nbr var-degree via clauses, ...]
    pos, neg = fv[:, :, 1], fv[:, :, 2]
    tot = jnp.maximum(pos + neg, 1e-6)
    # clause polarity of neighbours: for each var, mean over incident clauses of (#pos lits / 4)
    c_pos = Ip.sum(-1) / 4.0  # (B,m)
    c_neg = In.sum(-1) / 4.0
    nbr_cpos = (jnp.einsum("bcv,bc->bv", Ip + In, c_pos)) / deg
    # mean degree of variables co-appearing in a clause with v (approx via clause->var)
    c_meandeg = jnp.einsum("bcv,bv->bc", Ip + In, deg) / 4.0  # (B,m)
    nbr_meandeg = jnp.einsum("bcv,bc->bv", Ip + In, c_meandeg) / deg
    ones_v = jnp.ones((B, n), jnp.float32)
    fv_h = jnp.stack([ones_v, pos, neg, pos / tot, nbr_cpos, nbr_meandeg / 8.0,
                      (Ip.sum(1) + In.sum(1)) / (2.0 * m / n),  # raw deg / expected
                      jnp.zeros((B, n))], -1)  # pad slot
    # clause: [1, frac_pos, frac_neg, mean var pos_deg, mean var neg_deg, mean var tot_deg/8]
    ones_c = jnp.ones((B, m), jnp.float32)
    mean_pos = jnp.einsum("bcv,bv->bc", Ip + In, pos) / 4.0
    mean_neg = jnp.einsum("bcv,bv->bc", Ip + In, neg) / 4.0
    mean_deg = c_meandeg / 8.0
    fc_h = jnp.stack([ones_c, c_pos, c_neg, mean_pos, mean_neg, mean_deg], -1)
    return fv_h, fc_h


def init_backbone(rng, arch, H, rounds, FV):
    """Return backbone param dict (no heads, no SNAP)."""
    p = {}
    if arch in ("mp", "mp64", "mp128"):
        p = {"Ev": _W(rng, FV, H), "Ec": jnp.asarray(rng.normal(0, 1, (H,)).astype(np.float32)), "rounds": []}
        for _ in range(rounds):
            p["rounds"].append({k: _W(rng, H, H) for k in ["Uc", "Ap", "An", "Uv", "Bp", "Bn"]}
                               | {"bc": _zeros(H), "bv": _zeros(H)})
    elif arch == "attn":
        p = {"Ev": _W(rng, FV, H), "Ec": jnp.asarray(rng.normal(0, 1, (H,)).astype(np.float32)), "rounds": []}
        for _ in range(rounds):
            r = {k: _W(rng, H, H) for k in [
                "Uc", "Uv",
                "Qcp", "Kvp", "Vvp", "Qcn", "Kvn", "Vvn",  # clause attends to vars (+/-)
                "Qvp", "Kcp", "Vcp", "Qvn", "Kcn", "Vcn",  # var attends to clauses (+/-)
            ]}
            r["bc"] = _zeros(H); r["bv"] = _zeros(H)
            p["rounds"].append(r)
    elif arch == "tf":
        p = {"Ev": _W(rng, FV, H), "Ec0": jnp.asarray(rng.normal(0, 1, (H,)).astype(np.float32)), "layers": []}
        for _ in range(rounds):
            p["layers"].append({
                "Q": _W(rng, H, H), "K": _W(rng, H, H), "V": _W(rng, H, H), "O": _W(rng, H, H),
                "F1": _W(rng, H, 2 * H), "F2": _W(rng, 2 * H, H),
                "b1": _zeros(2 * H), "b2": _zeros(H),
            })
        p["V1"] = _W(rng, H, H); p["vb"] = _zeros(H)
    elif arch == "ds":
        p = {
            "Ev": _W(rng, FV, H), "Ec": jnp.asarray(rng.normal(0, 1, (H,)).astype(np.float32)),
            "Mv1": _W(rng, 2 * H, H), "Mv2": _W(rng, H, H), "bv1": _zeros(H), "bv2": _zeros(H),
            "Mc1": _W(rng, 2 * H, H), "Mc2": _W(rng, H, H), "bc1": _zeros(H), "bc2": _zeros(H),
        }
    elif arch == "mlp":
        p = {
            "Ev": _W(rng, FVH, H), "Ec": _W(rng, FCH, H),
            "Mv1": _W(rng, H, H), "Mv2": _W(rng, H, H), "bv1": _zeros(H), "bv2": _zeros(H),
            "Mc1": _W(rng, H, H), "Mc2": _W(rng, H, H), "bc1": _zeros(H), "bc2": _zeros(H),
        }
    else:
        raise ValueError(f"unknown arch {arch}")
    return p


def run_backbone(p, G, arch, H):
    """-> hv (B,n,H), hc (B,m,H)"""
    Ip, In = G["Ip"], G["In"]
    B, m, n = Ip.shape
    if arch in ("mp", "mp64", "mp128"):
        hv = jnp.tanh(G["fv"] @ p["Ev"])
        hc = jnp.broadcast_to(jnp.tanh(p["Ec"]), (B, m, H))
        for r in p["rounds"]:
            ac = (jnp.einsum("bcv,bvh->bch", Ip, hv @ r["Ap"]) + jnp.einsum("bcv,bvh->bch", In, hv @ r["An"])) / 4.0
            hc = hc + jnp.tanh(hc @ r["Uc"] + ac + r["bc"])
            av = (jnp.einsum("bcv,bch->bvh", Ip, hc @ r["Bp"]) + jnp.einsum("bcv,bch->bvh", In, hc @ r["Bn"])) / G["deg"][..., None]
            hv = hv + jnp.tanh(hv @ r["Uv"] + av + r["bv"])
        return hv, hc

    if arch == "attn":
        scale = jnp.float32(H ** -0.5)
        hv = jnp.tanh(G["fv"] @ p["Ev"])
        hc = jnp.broadcast_to(jnp.tanh(p["Ec"]), (B, m, H))
        for r in p["rounds"]:
            # clause attends to neighbouring vars
            qp = hc @ r["Qcp"]; kn = hv @ r["Kvp"]; vv = hv @ r["Vvp"]
            sp = jnp.einsum("bch,bvh->bcv", qp, kn) * scale
            ap = _masked_softmax(sp, Ip, axis=-1)
            mp = jnp.einsum("bcv,bvh->bch", ap, vv)
            qn = hc @ r["Qcn"]; kn = hv @ r["Kvn"]; vv = hv @ r["Vvn"]
            sn = jnp.einsum("bch,bvh->bcv", qn, kn) * scale
            an = _masked_softmax(sn, In, axis=-1)
            mn = jnp.einsum("bcv,bvh->bch", an, vv)
            hc = hc + jnp.tanh(hc @ r["Uc"] + mp + mn + r["bc"])
            # var attends to neighbouring clauses
            qp = hv @ r["Qvp"]; kc = hc @ r["Kcp"]; vc = hc @ r["Vcp"]
            sp = jnp.einsum("bvh,bch->bvc", qp, kc) * scale
            # Ip is (B,m,n) -> transpose mask to (B,n,m)
            ap = _masked_softmax(sp, jnp.transpose(Ip, (0, 2, 1)), axis=-1)
            mp = jnp.einsum("bvc,bch->bvh", ap, vc)
            qn = hv @ r["Qvn"]; kc = hc @ r["Kcn"]; vc = hc @ r["Vcn"]
            sn = jnp.einsum("bvh,bch->bvc", qn, kc) * scale
            an = _masked_softmax(sn, jnp.transpose(In, (0, 2, 1)), axis=-1)
            mn = jnp.einsum("bvc,bch->bvh", an, vc)
            hv = hv + jnp.tanh(hv @ r["Uv"] + (mp + mn) / G["deg"][..., None] + r["bv"])
        return hv, hc

    if arch == "tf":
        # clause tokens from signed incident var embeds
        ve = jnp.tanh(G["fv"] @ p["Ev"])  # (B,n,H)
        hc = (jnp.einsum("bcv,bvh->bch", Ip, ve) - jnp.einsum("bcv,bvh->bch", In, ve)) / 4.0
        hc = hc + jnp.tanh(p["Ec0"])
        for lyr in p["layers"]:
            q, k, v = hc @ lyr["Q"], hc @ lyr["K"], hc @ lyr["V"]
            attn = jax_softmax(jnp.einsum("bmh,bnh->bmn", q, k) * jnp.float32(H ** -0.5), axis=-1)
            hc = hc + jnp.tanh((attn @ v) @ lyr["O"])
            hc = hc + jnp.tanh((jnp.tanh(hc @ lyr["F1"] + lyr["b1"]) @ lyr["F2"]) + lyr["b2"])
        # vars pool from incident clauses
        hv = (jnp.einsum("bcv,bch->bvh", Ip, hc) + jnp.einsum("bcv,bch->bvh", In, hc)) / G["deg"][..., None]
        hv = hv + jnp.tanh(hv @ p["V1"] + p["vb"])
        return hv, hc

    if arch == "ds":
        hv0 = jnp.tanh(G["fv"] @ p["Ev"])  # (B,n,H)
        hc0 = jnp.broadcast_to(jnp.tanh(p["Ec"]), (B, m, H))
        g = jnp.concatenate([hv0.mean(1), hc0.mean(1)], -1)  # (B,2H) — but each MLP wants [local, global_same_dim]
        # use mean of all node embeds as global H-vector
        gH = 0.5 * (hv0.mean(1) + hc0.mean(1))  # (B,H)
        gHb = jnp.broadcast_to(gH[:, None, :], (B, n, H))
        gHc = jnp.broadcast_to(gH[:, None, :], (B, m, H))
        hv = jnp.tanh(jnp.concatenate([hv0, gHb], -1) @ p["Mv1"] + p["bv1"])
        hv = hv + jnp.tanh(hv @ p["Mv2"] + p["bv2"])
        hc = jnp.tanh(jnp.concatenate([hc0, gHc], -1) @ p["Mc1"] + p["bc1"])
        hc = hc + jnp.tanh(hc @ p["Mc2"] + p["bc2"])
        return hv, hc

    if arch == "mlp":
        fv_h, fc_h = handcraft(G)
        hv = jnp.tanh(fv_h @ p["Ev"])
        hv = hv + jnp.tanh(hv @ p["Mv1"] + p["bv1"])
        hv = hv + jnp.tanh(hv @ p["Mv2"] + p["bv2"])
        hc = jnp.tanh(fc_h @ p["Ec"])
        hc = hc + jnp.tanh(hc @ p["Mc1"] + p["bc1"])
        hc = hc + jnp.tanh(hc @ p["Mc2"] + p["bc2"])
        return hv, hc

    raise ValueError(f"unknown arch {arch}")

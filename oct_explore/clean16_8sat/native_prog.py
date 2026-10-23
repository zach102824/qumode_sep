"""Evolvable NETWORK (8-SAT port of clean16/native_prog.py = net16r ckpt_110).
train(api, budget_s) -> predict;  predict(n, V, SG, m, place)
V (F,m,K) clause variable indices (K=8 for paper 8-SAT; padded slots have SG=0),
SG (F,m,K) literal signs (+1/-1; 0 = pad). Same GNN / dS / dB / grover families as ckpt_110.
Deviations from 4-SAT ckpt_110 are documented in NOTES.md (k-generalised geom/local_tables/wc;
size-free /K norms; sparse circuit_p is in the driver, not here).
"""
import time, numpy as np, jax, jax.numpy as jnp
SEED = 0  # clean16_8sat: set by driver

# EVOLVE-BLOCK-START
# FAMILY: grover - role-staged capped Grover + hop-2 demotion (L1 opt, L2-3 π, L4 graded)
# parent: (net16o ckpt_60 best fd65b9be: selection 56.9/599, fair TEB n=16 84.4/5229)
# 8-SAT adaptations: V/SG last dim K (=8); SUB enumeration replaced by closed-form cavity counts;
# local_tables uses incidence (no (m,K) one_hot); /4 -> /K size-free; wc (H,4) layers unchanged.
H = 32
LAM_GLOB = 0.0
TAIL_W = 1.0
G0 = np.array([-3.1275, -6.49, 9.6225, 8.4856], np.float32); B0 = np.array([1.1305, 0.3672, -0.5844, -0.3928], np.float32)

def features(n, V, SG, m):
    """unsigned incidence (even) + signed incidence (odd). Pad slots (SG==0) contribute 0."""
    F, _, K = V.shape; inc = np.zeros((F, m, n), np.float32); sgn = np.zeros((F, m, n), np.float32)
    fi = np.arange(F)[:, None, None]; ci = np.arange(m)[None, :, None]
    np.add.at(inc, (fi, ci, V), (SG != 0).astype(np.float32)); np.add.at(sgn, (fi, ci, V), np.asarray(SG, np.float32))
    deg = inc.sum(1); mr = np.maximum((inc.sum(-1) > 0).sum(1, keepdims=True), 1.0); sc = (0.5 * K) * mr / n
    xe = np.stack([np.ones_like(deg), deg / sc], -1)
    xo = (sgn.sum(1) / sc)[..., None]
    return jnp.asarray(xe), jnp.asarray(xo), jnp.asarray(inc), jnp.asarray(sgn), jnp.asarray(V, jnp.int32)

def init(rng):
    def W(i, o, s=1.0): return jnp.asarray(rng.normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    return dict(We=W(2, H), Wo=W(1, H), Wce=W(H, H), Wco=W(H, H), Wve=W(H, H), Wvo=W(H, H),
                Wc=W(2 * H, H), wc=W(H, 4, 0.05), wp=W(H, 1, 0.5)[:, 0],
                g=jnp.asarray(G0), b=jnp.asarray(B0), Wg=W(H, 8, 0.01),
                As=W(H, H), asl=jnp.zeros((4, H)), Ab=W(H, H), bsl=jnp.zeros((4, H)),
                Wq1=W(H + 4, H), Wq2=jnp.zeros((H, 4)), Wgg=jnp.zeros((4, 8)), Lv=W(1, H, 0.3)[0],
                Wch=W(H, H, 0.3), Wpc=W(2 * H + 3, H, 0.3), bpl=W(4, H, 0.05),
                wsk=W(1, 4, 0.05)[0], rs=W(1, 4, 0.05)[0], rb=W(1, 4, 0.05)[0])

def geom(V, place, n, cm=None):
    """size-free chain geometry. Closed-form: lost = fraction of nonempty literal subsets whose
    cavity support is NOT one cavity or a neighbour pair (= 1 - rcnt/(2^K-1)), matching SUB
    enumeration of the k=4 program without building (F,R,m,2^K-1) tensors."""
    cav = jnp.asarray(place["cav"]); F, R, C, _ = cav.shape; m, K = V.shape[1], V.shape[2]
    eq = (cav[..., None] == jnp.arange(n)).astype(jnp.float32)
    cv = jnp.einsum("frcjn,c->frn", eq, jnp.arange(C, dtype=jnp.float32))
    c = jnp.take_along_axis(jnp.broadcast_to(cv[:, :, None, :], (F, R, m, n)), jnp.broadcast_to(V[:, None], (F, R, m, K)), axis=-1)
    na = (c[..., None] == jnp.arange(C, dtype=jnp.float32)).sum(-2).astype(jnp.float32)  # (F,R,m,C) #lits in cavity
    # # nonempty subsets living in one cavity or a neighbour pair = sum_c (2^{n_c}-1) + sum_p (2^{n_p+n_{p+1}}-1) - sum_interior (2^{n_c}-1)
    rcnt = (jnp.exp2(na[..., :-1] + na[..., 1:]) - 1.0).sum(-1) - (jnp.exp2(na[..., 1:-1]) - 1.0).sum(-1)
    lost = 1.0 - rcnt / float((1 << K) - 1)
    span = (na > 0).sum(-1).astype(jnp.float32) / float(K)
    hop2 = ((c.max(-1) - c.min(-1)) > 1.5).astype(jnp.float32)
    rep_full = 1.0 - hop2
    if cm is not None:
        cm_ = cm[:, None, :]
        lost, rep_full, span, hop2 = lost * cm_, rep_full * cm_, span * cm_, hop2 * cm_
    return jnp.stack([lost, rep_full, span, hop2], -1), lost, c

def sign_table(k):
    i = np.arange(32)[:, None]; j = np.arange(5)[None]; b = (i >> np.maximum(k - 1 - j, 0)) & 1
    return jnp.asarray(np.where((j < k) & (i < (1 << k)), 1.0 - 2.0 * b, 0.0).astype(np.float32))

def corr(p, he, ho, place, lv, lvs, lost, cidx, K=8.0, mr=None):
    """size-free tables. /4 -> /K; .mean(2) -> .sum(2)/mr so pad slots (lost=0) do not dilute. mr: (F,) or scalar."""
    cav = jnp.asarray(place["cav"]); ks = place["ks"]; F, R, C, _ = cav.shape; ST = sign_table(ks[0]); m = cidx.shape[2]
    if mr is None: mr = jnp.full((F,), float(m), jnp.float32)
    mr3 = mr[:, None, None]  # (F,1,1) for (F,R,*) broadcasts
    ix = jnp.maximum(cav, 0).reshape(F, R, C * 5); msk = (cav >= 0)[..., None]
    lvc = jnp.take_along_axis(lv, ix, axis=-1).reshape(F, R, C, 5)
    lsc = jnp.take_along_axis(lvs, ix, axis=-1).reshape(F, R, C, 5)
    take = lambda h: jax.vmap(lambda u, c: u[jnp.maximum(c, 0)])(h, cav.reshape(F, -1)).reshape(F, R, C, 5, -1)
    te, to = take(he), take(ho)
    hcv = (te + to + lvc[..., None] * p["Lv"]) * msk
    h0 = hcv.sum(3) / (msk.sum(3) + 1e-6); hcav = h0
    for _ in range(2):
        left = jnp.concatenate([jnp.zeros_like(hcav[:, :, :1]), hcav[:, :, :-1]], 2)
        right = jnp.concatenate([hcav[:, :, 1:], jnp.zeros_like(hcav[:, :, :1])], 2)
        hcav = jnp.tanh(hcav + (left + right) @ p["Wch"])
    hcav = hcav + h0
    hcv = hcv + hcav[:, :, :, None, :] * msk
    z = jnp.einsum("kj,frcjh->frckh", ST, hcv)
    us, ub = jnp.tanh(z @ p["As"]), jnp.tanh(z @ p["Ab"])
    dS = jnp.einsum("frckh,lh->frlck", us, p["asl"])
    dB = jnp.einsum("frakh,frbqh,lh->frlakq", ub[:, :, :-1], ub[:, :, 1:], p["bsl"])
    clo, chi = cidx.min(-1), cidx.max(-1); hop2 = (chi - clo) > 1.5
    pidx = jnp.arange(C - 1, dtype=jnp.float32); cpos = jnp.arange(C, dtype=jnp.float32)
    onpath = ((clo[..., None] <= pidx) & (chi[..., None] > pidx) & hop2[..., None]).astype(jnp.float32)
    sl = (lost[..., None] * onpath).sum(2) / mr3
    wt = (lost * hop2)[..., None]
    nL = (cidx[..., None] == pidx).sum(-2).astype(jnp.float32)
    nR = (cidx[..., None] == (pidx + 1)).sum(-2).astype(jnp.float32)
    pf = jnp.stack([(wt * nL).sum(2) / (mr3 * K), (wt * nR).sum(2) / (mr3 * K), sl], -1)
    hp = jnp.tanh(jnp.concatenate([hcav[:, :, :-1], hcav[:, :, 1:], pf], -1) @ p["Wpc"])
    dB = dB + jnp.einsum("frph,lh,frakh,frbqh,frp->frlakq", hp, p["bpl"], ub[:, :, :-1], ub[:, :, 1:], sl)
    odd = (to * msk).sum(-1)
    ismid = ((clo[..., None] < cpos) & (chi[..., None] > cpos) & hop2[..., None]).astype(jnp.float32)
    mm = (lost[..., None] * ismid).sum(2) / mr3
    dS = dS + jnp.einsum("kj,frcj,l->frlck", ST, lvc * jnp.tanh(odd) + mm[..., None], p["rs"])
    zL = jnp.einsum("kj,frcj->frck", ST, lsc[:, :, :-1]); zR = jnp.einsum("kj,frcj->frck", ST, lsc[:, :, 1:])
    dB = dB + jnp.einsum("frp,l,frpk,frpq->frlpkq", sl, p["rb"], zL, zR)
    return dS, dB

FAMILIES = ("grover",)
ROLE = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]]
NAT_TAU = 0.5
GROVER_QMAX = 0.25
def native_init(rng):
    W = lambda i, o, s: jnp.asarray(rng.normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    return dict(nr1=W(H + 2, 16, 0.5), nr2=jnp.zeros((16,), jnp.float32), nrb=jnp.asarray(1.0, jnp.float32),
                gS=jnp.zeros((4, 4), jnp.float32), gB=jnp.zeros((4, 4), jnp.float32), rq=jnp.asarray(0.0, jnp.float32),
                Wh=jnp.zeros((3, 4), jnp.float32),
                gopt=jnp.zeros((4,), jnp.float32),
                ghop=jnp.asarray(0.0, jnp.float32),
                gfin=jnp.asarray(0.0, jnp.float32),
                bref=jnp.zeros((4,), jnp.float32))

def local_tables(p, hc, V, inc, sgn, place, n, hop2, mr):
    """per-cavity clause-local tables via incidence (no one_hot over K slots).
    Pad slots (inc=0) contribute neither to Lc nor to o; keep uses o/K so the nr1/nrb scale
    matches the k=4 seed (where o/4 was the in-scope fraction)."""
    cav = jnp.asarray(place["cav"]); F, R, C, _ = cav.shape; k = place["ks"][0]; K = float(V.shape[2])
    eq = (cav[..., None] == jnp.arange(n)).astype(jnp.float32)                          # (F,R,C,5,n)
    i = np.arange(32)[:, None]; j = np.arange(5)[None]
    BT = jnp.asarray(np.where(j < k, (i >> np.maximum(k - 1 - j, 0)) & 1, 0).astype(np.float32))
    B1 = jnp.einsum("frcjn,ij->frcin", eq, BT); B0 = jnp.einsum("frcjn,ij->frcin", eq, 1.0 - BT)
    cpos = 0.5 * (inc + sgn); cneg = 0.5 * (inc - sgn)                                  # (F,m,n) polarity counts
    # Lc[f,r,m,c,i] = 1 iff every in-cavity literal of clause m is falsified at level i
    Lc = ((jnp.einsum("fmn,frcin->frmci", cpos, B1) + jnp.einsum("fmn,frcin->frmci", cneg, B0)) < 0.5).astype(jnp.float32)
    cnt = jnp.einsum("frcjn,fmn->frmc", eq, inc); o = K - cnt
    # keep = 1 - rescue; o/K replaces o/4 so nr1/nrb scale is size-free across k
    h_part = (hc @ p["nr1"][:H])[:, None, :, None, :]
    o_part = jnp.stack([o / K, (o > 0).astype(jnp.float32)], -1) @ p["nr1"][H:]
    kp = 1.0 - jax.nn.sigmoid(jnp.tanh(h_part + o_part) @ p["nr2"] + p["nrb"] * 4.0 * (o / K - 0.5))
    valid = jnp.asarray((np.arange(32) < (1 << k)).astype(np.float32))
    nm = (n / mr)[:, None, None]                                                        # (F,1,1); broadcasts over trailing dims of (F,R,C,32) via einsum path
    E1 = jnp.einsum("frmc,frmci->frci", (cnt > 0) * kp, Lc) * valid * nm[..., None]
    hop_pen = jnp.einsum("frm,frmc,frmci->frci", hop2, (cnt > 0).astype(jnp.float32), 1.0 - Lc) * valid * nm[..., None]
    return E1, hop_pen, valid

def soft_rank(E, valid, t=0.3):
    d = (E[..., None, :] - E[..., :, None]) / t
    return (jax.nn.sigmoid(-d) * valid).sum(-1) / valid.sum() * valid

def native(p, hc, V, inc, sgn, place, n, dS, dB, hard, hop2, mr):
    E1, hop_pen, valid = local_tables(p, hc, V, inc, sgn, place, n, hop2, mr)
    on = jnp.float32(1.0 if "grover" in FAMILIES else 0.0)
    mod = 1.0 + jnp.tanh(hard @ p["Wh"])
    E = E1 + p["ghop"] * hop_pen
    r = soft_rank(E, valid)
    q = GROVER_QMAX * jax.nn.sigmoid(p["rq"])
    mark = jax.nn.sigmoid((q - r) / 0.03) * valid
    grade = jnp.clip((q - r) / (q + 1e-6), 0.0, 1.0) * valid
    grole = on * jnp.asarray(ROLE[3], jnp.float32)
    m1 = jnp.asarray([1.0, 0.0, 0.0, 0.0], jnp.float32)
    m4 = jnp.asarray([0.0, 0.0, 0.0, 1.0], jnp.float32)
    dS = dS + jnp.einsum("f,l,frci->frlci", mod[:, 0], on * p["gopt"] * m1, E1)
    dS = dS + jnp.einsum("f,l,frci->frlci", mod[:, 3], p["gS"][3] * grole, jnp.pi * mark)
    dS = dS + jnp.einsum("f,l,frci->frlci", mod[:, 3], on * p["gfin"] * m4, jnp.pi * grade)
    return dS, dB

def apply(p, xe, xo, inc, sgn, V, place=None, cm=None):
    """3-round shared-weight MP. /4 -> /K (K = V.shape[2]). cm (F,m) real-clause mask for pad slots."""
    K = float(V.shape[2]); e = jnp.tanh(xe @ p["We"]); o = jnp.tanh(xo @ p["Wo"]); vd = inc.sum(1)[..., None] + 1.0
    for _ in range(4):
        me = jnp.einsum("fmn,fnh->fmh", inc, e); mo = jnp.einsum("fmn,fnh->fmh", sgn, o)
        hc = jnp.tanh((me @ p["Wce"] + mo @ p["Wco"]) / K)
        e = jnp.tanh(e + (jnp.einsum("fmn,fmh->fnh", inc, hc) / vd) @ p["Wve"])
        o = jnp.tanh(o + (jnp.einsum("fmn,fmh->fnh", sgn, hc) / vd) @ p["Wvo"])
    hc = jnp.tanh(jnp.concatenate([jnp.einsum("fmn,fnh->fmh", inc, e), jnp.einsum("fmn,fnh->fmh", sgn, o)], -1) / K @ p["Wc"])
    mr = cm.sum(1) if cm is not None else jnp.full((V.shape[0],), float(V.shape[1]), jnp.float32)  # (F,)
    if place is not None:
        gr, lost, cidx = geom(V, place, inc.shape[2], cm=cm); geo = gr.sum(1) / jnp.maximum(mr[:, None, None], 1.0)
        lv = jnp.einsum("frm,fmn->frn", lost, inc) / vd[:, None, :, 0]
        lvs = jnp.einsum("frm,fmn->frn", lost * gr[..., 3], sgn) / vd[:, None, :, 0]
    else:
        geo = jnp.zeros(V.shape[:2] + (4,), jnp.float32); lost = cidx = lvs = None
    gb = jnp.tanh(e.mean(1) @ p["Wg"] + geo.mean(1) @ p["Wgg"])
    w = hc @ p["wc"] + jnp.tanh(jnp.concatenate([hc, geo], -1) @ p["Wq1"]) @ p["Wq2"] + geo[..., 3, None] * p["wsk"]
    if cm is not None: w = w * cm[..., None]
    bref = p["bref"] * jnp.asarray([0.0, 0.0, 0.0, 1.0], jnp.float32)
    out = dict(gamma=p["g"][None] + gb[:, :4], beta=p["b"][None] + gb[:, 4:] + bref[None], w=w, pv=jax.nn.sigmoid(o @ p["wp"]))
    if place is not None:
        out["dS"], out["dB"] = corr(p, e, o, place, lv, lvs, lost, cidx, K=K, mr=mr)
        out["dS"], out["dB"] = native(p, hc, V, inc, sgn, place, inc.shape[2], out["dS"], out["dB"], geo.mean(1)[:, :3], gr[..., 3], mr)
    return out

def train(api, budget_s):
    t0 = time.time(); rng = np.random.default_rng(SEED); p = init(rng); p.update(native_init(rng))

    from jax.flatten_util import ravel_pytree
    th, unr = ravel_pytree(p); mo = jnp.zeros_like(th); ve = jnp.zeros_like(th)
    data = {n: api.data(n) for n in api.n_list}
    feats = {}; cms = {}
    for n, d in data.items():
        fx = features(n, d["V"], d["SG"], d["m"]); feats[n] = fx
        cms[n] = jnp.asarray((np.asarray(d["SG"]) != 0).any(-1).astype(np.float32))   # (F,m) real-clause mask
    # smaller batches at k=8 (m≈1400/2100 vs 79/119): keep peak RAM under ~4 GB
    bs = {8: 4, 12: 2}
    def loss(th, n, fx, pl, b, cm):
        pp = unr(th); out = apply(pp, *fx, place=pl, cm=cm)
        P = api.p_from(n, out, b); nl = -jnp.log(P.mean(1) + 1e-12)
        xe, xo, inc, sgn, V = fx; of = apply(pp, xe, -xo, inc, -sgn, V, place=pl, cm=cm)
        pen = ((out["gamma"] - of["gamma"]) ** 2).mean() + ((out["beta"] - of["beta"]) ** 2).mean()
        pen = pen + ((out["w"] - of["w"]) ** 2).mean() + ((out["pv"] + of["pv"] - 1.0) ** 2).mean()
        _, Lst, _ = geom(V, pl, inc.shape[2], cm=cm)
        mr_ = jnp.maximum(cm.sum(1), 1.0)
        lw = 1.0 + 1.5 * jax.lax.stop_gradient((Lst.sum((1, 2)) / mr_))
        kc = max(1, int(np.ceil(0.1 * nl.shape[0])))
        tot = 0.5 * (nl * lw).mean() + 0.5 * jnp.median(nl) + TAIL_W * jax.lax.top_k(nl, kc)[0].mean() + 0.05 * pen
        return tot
    vg = {n: jax.jit(jax.value_and_grad(lambda th, fx, pl, b, cm, n=n: loss(th, n, fx, {"cav": pl, "ks": api.place(n, [0])["ks"]}, b, cm))) for n in api.n_list}
    t = 0; T = budget_s * 0.9; ema = None
    while time.time() - t0 < T:
        g = 0
        for n in api.n_list:
            idx = rng.choice(data[n]["F"], bs.get(n, 2), replace=False); fx = tuple(x[idx] for x in feats[n])
            _, gn = vg[n](th, fx, jnp.asarray(api.place(n, idx)["cav"]), api.batch(n, idx), cms[n][idx]); g = g + gn
        gnorm = float(jnp.sqrt((g * g).sum()))
        if not np.isfinite(gnorm): continue
        t += 1; g = g * min(1.0, 10.0 / (gnorm + 1e-9))
        fr = (time.time() - t0) / T; lr = 3e-3 * (1.0 if fr < 0.7 else max(0.2, (1.0 - fr) / 0.3))
        mo = 0.9 * mo + 0.1 * g; ve = 0.999 * ve + 0.001 * g * g
        th = th - lr * (mo / (1 - 0.9 ** t)) / (jnp.sqrt(ve / (1 - 0.999 ** t)) + 1e-8)
        if fr > 0.4: ema = th if ema is None else 0.98 * ema + 0.02 * th
    pf = unr(th if ema is None or not bool(jnp.all(jnp.isfinite(ema))) else ema)
    def predict(n, V, SG, m, place=None):
        V, SG = np.asarray(V), np.asarray(SG); res = []; cm_all = (SG != 0).any(-1).astype(np.float32)
        for s in range(0, V.shape[0], 8):
            pl = None if place is None else dict(cav=np.asarray(place["cav"])[s:s + 8], ks=place["ks"])
            out = apply(pf, *features(n, V[s:s + 8], SG[s:s + 8], m), place=pl, cm=jnp.asarray(cm_all[s:s + 8]))
            res.append({k: np.nan_to_num(np.asarray(v)) for k, v in out.items()})
        return {k: np.concatenate([r[k] for r in res]) for k in res[0]}
    return predict
# EVOLVE-BLOCK-END

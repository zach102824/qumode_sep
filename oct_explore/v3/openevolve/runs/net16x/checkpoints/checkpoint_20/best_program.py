"""Evolvable NETWORK: formula -> (gamma_l, beta_l, w_jl, p_v) for the locked native cavity-chain circuit.
train(api, budget_s) -> predict;  predict(n, V, SG, m, place)
(+ OPTIONAL dS (F,R,L,C,32) SNAP and dB (F,R,L,C-1,32,32) neighbour-bus additive phase tables; api.place(n, idx) gives cavity contents) -> dict(gamma (F,L), beta (F,L), w (F,m,L), pv (F,n)).
V (F,m,4) clause variable indices, SG (F,m,4) literal signs (+1/-1; literal violated when x_v == (0 if s>0 else 1)).
api.data(n) -> dict(V, SG, m, F) training formulas (n in api.n_list = [10, 15]);
api.batch(n, fidx) -> static circuit inputs; api.p_from(n, outs, batch) -> differentiable p (F, R) over R placements (col 0 = min-cut).
Fair objective = round-robin: mean over the R placements. Allowed imports: numpy, jax, math, functools, itertools, collections, time.
"""
import time, numpy as np, jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
# FAMILY: grover - Grover-like reflections: local clause-rank oracles + pair diffusion reflections (pair unitaries)
# parent: (net16o ckpt_60 best fd65b9be: selection 56.9/599, fair TEB n=16 84.4/5229)
H = 32
LAM_GLOB = 0.0   # global-cost imitation is NOT the goal; kept available but off
TAIL_W = 1.0     # weight of the CVaR-10% tail term in the loss
G0 = np.array([-3.1275, -6.49, 9.6225, 8.4856], np.float32); B0 = np.array([1.1305, 0.3672, -0.5844, -0.3928], np.float32)  # F32_gray_chain (trained n=10+15 only)

def features(n, V, SG, m):
    """unsigned incidence (even) + signed incidence (odd). Flipping SG on var i flips xo_i and sgn[..., i]."""
    F = V.shape[0]; inc = np.zeros((F, m, n), np.float32); sgn = np.zeros((F, m, n), np.float32)
    fi = np.arange(F)[:, None, None]; ci = np.arange(m)[None, :, None]
    np.add.at(inc, (fi, ci, V), 1.0); np.add.at(sgn, (fi, ci, V), np.asarray(SG, np.float32))
    deg = inc.sum(1); sc = 2.0 * m / n
    xe = np.stack([np.ones_like(deg), deg / sc], -1)
    xo = (sgn.sum(1) / sc)[..., None]
    return jnp.asarray(xe), jnp.asarray(xo), jnp.asarray(inc), jnp.asarray(sgn), jnp.asarray(V, jnp.int32)

def init(rng):
    def W(i, o, s=1.0): return jnp.asarray(rng.normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    return dict(We=W(2, H), Wo=W(1, H), Wce=W(H, H), Wco=W(H, H), Wve=W(H, H), Wvo=W(H, H),
                Wc=W(2 * H, H), wc=W(H, 4, 0.05), wp=W(H, 1, 0.5)[:, 0],
                g=jnp.asarray(G0), b=jnp.asarray(B0), Wg=W(H, 8, 0.01),
                As=W(H, H), asl=jnp.zeros((4, H)), Ab=W(H, H), bsl=jnp.zeros((4, H)),  # correction heads; zero-init => starts at seed
                Wq1=W(H + 4, H), Wq2=jnp.zeros((H, 4)), Wgg=jnp.zeros((4, 8)), Lv=W(1, H, 0.3)[0],
                Wch=W(H, H, 0.3), Wpc=W(2 * H + 3, H, 0.3), bpl=W(4, H, 0.05),  # chain MP + pair head (skip-gated)
                wsk=W(1, 4, 0.05)[0], rs=W(1, 4, 0.05)[0], rb=W(1, 4, 0.05)[0])  # hop-2 w / lost-gated table residuals

SUB = np.array([[(s >> i) & 1 for i in range(4)] for s in range(1, 16)], bool)  # 15 nonempty literal subsets, last = full clause

def geom(V, place, n):
    """size-free chain geometry per clause/placement: fraction of Walsh subsets not inside one cavity or neighbour pair."""
    cav = jnp.asarray(place["cav"]); F, R, C, _ = cav.shape; m = V.shape[1]
    eq = (cav[..., None] == jnp.arange(n)).astype(jnp.float32)
    cv = jnp.einsum("frcjn,c->frn", eq, jnp.arange(C, dtype=jnp.float32))
    c = jnp.take_along_axis(jnp.broadcast_to(cv[:, :, None, :], (F, R, m, n)), jnp.broadcast_to(V[:, None], (F, R, m, 4)), axis=-1)
    cs = c[..., None, :]; M = jnp.asarray(SUB)
    mx = jnp.max(jnp.where(M, cs, -1.0), -1); mn = jnp.min(jnp.where(M, cs, 99.0), -1)
    rep = (mx - mn <= 1.5).astype(jnp.float32); lost = 1.0 - rep.mean(-1)
    span = (c[..., None] == jnp.arange(C, dtype=jnp.float32)).any(-2).sum(-1) / 4.0
    hop2 = ((c.max(-1) - c.min(-1)) > 1.5).astype(jnp.float32)  # skip-cavity clauses (lost Walsh); 0 on n=8
    return jnp.stack([lost, rep[..., -1], span, hop2], -1), lost, c

def sign_table(k):
    """(32,5): +-1 digit pattern of SNAP/bus table index (MSB first) for a k-bit cavity, 0 for unused slots/indices."""
    i = np.arange(32)[:, None]; j = np.arange(5)[None]; b = (i >> np.maximum(k - 1 - j, 0)) & 1
    return jnp.asarray(np.where((j < k) & (i < (1 << k)), 1.0 - 2.0 * b, 0.0).astype(np.float32))

def corr(p, he, ho, place, lv, lvs, lost, cidx):
    """size-free tables: even he is SG-invariant; odd ho flips with SG, mixed with ST (Z digit pattern).
    Signed hop-2 bus residual uses literal polarity (s_i s_j Z_i Z_j); middle SNAP fires on skip-over cavities
    (lvc=0 there). 3 hops only when C>3 so n=16 ends meet without extra smoothing on n=8/12."""
    cav = jnp.asarray(place["cav"]); ks = place["ks"]; F, R, C, _ = cav.shape; ST = sign_table(ks[0])
    ix = jnp.maximum(cav, 0).reshape(F, R, C * 5); msk = (cav >= 0)[..., None]
    lvc = jnp.take_along_axis(lv, ix, axis=-1).reshape(F, R, C, 5)
    lsc = jnp.take_along_axis(lvs, ix, axis=-1).reshape(F, R, C, 5)
    take = lambda h: jax.vmap(lambda u, c: u[jnp.maximum(c, 0)])(h, cav.reshape(F, -1)).reshape(F, R, C, 5, -1)
    te, to = take(he), take(ho)
    hcv = (te + to + lvc[..., None] * p["Lv"]) * msk
    h0 = hcv.sum(3) / (msk.sum(3) + 1e-6); hcav = h0
    for _ in range(2):  # same 2-hop depth at n=8/12/16 so Wch transfers
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
    sl = (lost[..., None] * onpath).mean(2)
    wt = (lost * hop2)[..., None]
    nL = (cidx[..., None] == pidx).sum(-2).astype(jnp.float32)
    nR = (cidx[..., None] == (pidx + 1)).sum(-2).astype(jnp.float32)
    pf = jnp.stack([(wt * nL).mean(2) / 4.0, (wt * nR).mean(2) / 4.0, sl], -1)
    hp = jnp.tanh(jnp.concatenate([hcav[:, :, :-1], hcav[:, :, 1:], pf], -1) @ p["Wpc"])
    dB = dB + jnp.einsum("frph,lh,frakh,frbqh,frp->frlakq", hp, p["bpl"], ub[:, :, :-1], ub[:, :, 1:], sl)
    odd = (to * msk).sum(-1)
    ismid = ((clo[..., None] < cpos) & (chi[..., None] > cpos) & hop2[..., None]).astype(jnp.float32)
    mm = (lost[..., None] * ismid).mean(2)
    dS = dS + jnp.einsum("kj,frcj,l->frlck", ST, lvc * jnp.tanh(odd) + mm[..., None], p["rs"])
    zL = jnp.einsum("kj,frcj->frck", ST, lsc[:, :, :-1]); zR = jnp.einsum("kj,frcj->frck", ST, lsc[:, :, 1:])
    dB = dB + jnp.einsum("frp,l,frpk,frpq->frlpkq", sl, p["rb"], zL, zR)
    return dS, dB

# ---------------- NATIVE PHASE PROGRAMS (framing: do NOT imitate the global cost; build local Fock-level phase programs) ----------------
# Families (gate index): 0 clause-local optimistic tables, 1 conditional pair messages (alternating chain sweep), 2 rank level shaping,
# 3 cavity-local Grover (pi on a capped soft candidate set). Gates gS/gB are zero-initialised => the seed starts at the parent's function.
# Hardness add-on (family 6): intensive formula features modulate each family's gate. Only dS/dB phase VALUES change (locked gate sequence).
FAMILIES = ("grover",)   # enabled native families on this island
ROLE = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]]  # L1 opt / off / L4 rank / L2-3 grover
NAT_TAU = 0.5      # temperature of the soft chain messages
GROVER_QMAX = 0.25 # candidate set capped at <= 25% of a cavity's levels (keeps it far from a pv-like single answer)
def native_init(rng):
    W = lambda i, o, s: jnp.asarray(rng.normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    # all grover-island gates start at 0 => exact parent circuit at step 0
    return dict(nr1=W(H + 2, 16, 0.5), nr2=jnp.zeros((16,), jnp.float32), nrb=jnp.asarray(1.0, jnp.float32),
                gS=jnp.zeros((4, 4), jnp.float32), gB=jnp.zeros((4, 4), jnp.float32), rq=jnp.asarray(0.0, jnp.float32),
                Wh=jnp.zeros((3, 4), jnp.float32),
                gopt=jnp.zeros((4,), jnp.float32),   # L1 optimistic SNAP
                ghop=jnp.asarray(0.0, jnp.float32),  # hop-2 energy penalty
                gfin=jnp.asarray(0.0, jnp.float32),  # L4 rank-graded SNAP
                bref=jnp.zeros((4,), jnp.float32))   # last-layer beta refocus
def local_tables(p, hc, V, sgn, place, n):
    """per-cavity clause-local tables E1 (F,R,C,32) and neighbour-pair tables Ep (F,R,C-1,32,32) from in-scope literals only,
    each clause discounted by a learned keep = 1 - rescue probability (out-of-scope literals may satisfy it)."""
    cav = jnp.asarray(place["cav"]); F, R, C, _ = cav.shape; k = place["ks"][0]; m = V.shape[1]
    lsg = jnp.take_along_axis(sgn, V, axis=2); vbit = (lsg < 0).astype(jnp.float32)                 # (F,m,4) violating value
    eq = (cav[..., None] == jnp.arange(n)).astype(jnp.float32).reshape(F, R, C, 5, n)
    A = jnp.einsum("frcjn,fmtn->frmtcj", eq, jax.nn.one_hot(V, n, dtype=jnp.float32))              # literal -> (cavity, slot)
    i = np.arange(32)[:, None]; j = np.arange(5)[None]
    BT = jnp.asarray(np.where(j < k, (i >> np.maximum(k - 1 - j, 0)) & 1, 0).astype(np.float32))   # (32,5) slot bit per level
    match = jnp.einsum("frmtcj,ij->frmtci", A, BT) * vbit[:, None, :, :, None, None] + \
            jnp.einsum("frmtcj,ij->frmtci", A, 1.0 - BT) * (1.0 - vbit)[:, None, :, :, None, None]
    inc = A.sum(-1)                                                                                 # (F,R,m,4,C)
    Lc = jnp.prod(1.0 - inc[..., None] + match, axis=3)                                             # (F,R,m,C,32)
    cnt = inc.sum(3); o = 4.0 - cnt                                                                  # in-scope count per cavity
    def keep(oo):
        x = jnp.concatenate([jnp.broadcast_to(hc[:, None, :, None, :], oo.shape + (H,)), (oo / 4.0)[..., None], (oo > 0)[..., None].astype(jnp.float32)], -1)
        return 1.0 - jax.nn.sigmoid(jnp.tanh(x @ p["nr1"]) @ p["nr2"] + p["nrb"] * (oo - 2.0))
    valid = jnp.asarray((np.arange(32) < (1 << k)).astype(np.float32))
    E1 = (n / m) * jnp.einsum("frmc,frmci->frci", (cnt > 0) * keep(o), Lc) * valid
    _, _, cidx = geom(V, place, n)
    hop2 = ((cidx.max(-1) - cidx.min(-1)) > 1.5).astype(jnp.float32)  # skip-cavity clauses (F,R,m)
    # (F,R,C,32): extra energy iff a level falsifies in-cavity literals of a hop-2 clause
    hop_pen = (n / m) * jnp.einsum("frm,frmc,frmci->frci", hop2, (cnt > 0).astype(jnp.float32), 1.0 - Lc) * valid
    return E1, hop_pen, valid
def chain_messages(E1, Ep, valid):
    """soft min-sum sweeps along the chain over local (optimistic) tables: fwd[c] = message INTO cavity c from the left, bwd from the right."""
    C = E1.shape[2]; big = 1e3 * (1.0 - valid); z = jnp.zeros_like(E1[:, :, 0])
    def norm(x): return (x - (x * valid).sum(-1, keepdims=True) / valid.sum()) * valid
    fwd = [z]
    for c in range(C - 1):
        x = (E1[:, :, c] + fwd[-1] + big)[..., :, None] + Ep[:, :, c]
        fwd.append(norm(-NAT_TAU * jax.nn.logsumexp(-x / NAT_TAU, axis=-2)))
    bwd = [z]
    for c in range(C - 1, 0, -1):
        x = (E1[:, :, c] + bwd[-1] + big)[..., None, :] + Ep[:, :, c - 1]
        bwd.append(norm(-NAT_TAU * jax.nn.logsumexp(-x / NAT_TAU, axis=-1)))
    return jnp.stack(fwd, 2), jnp.stack(bwd[::-1], 2)
def soft_rank(E, valid, t=0.3):
    d = (E[..., None, :] - E[..., :, None]) / t                                                    # [a, a'] = E(a') - E(a)
    return (jax.nn.sigmoid(-d) * valid).sum(-1) / valid.sum() * valid                              # fraction of levels below a
def native(p, hc, V, sgn, place, n, dS, dB, hard):
    """Role-staged grover on SNAP only (no pair tables): L1 broad optimistic dS;
    L2–L3 π on a ≤25% set; L4 rank-graded phase. Energy = E1 + hop-2 demotion."""
    E1, hop_pen, valid = local_tables(p, hc, V, sgn, place, n)
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

# ---------------- PAIR-UNITARY HEAD (net16x island: grover) ----------------
# Diffusion AFTER the mixer (slot 2) so L3–L4 can move the ~8 correlated cut bits the 1-bit mixer cannot.
# Shared A=1 (one expm / layer): (0) pair-uniform, (1) per-cavity uniform, (2) boundary-fiber on the
# 2 cut bits (LSB of left, MSB of right). Fiber is the new move; (0)(1) stay learnable from 0.
PAIR_SLOTS = (2,)
def _grover_gen():
    sA = np.full((16, 1), 0.25, np.float32); PA = sA @ sA.T; I = np.eye(16, dtype=np.float32)
    sAB = np.full((256, 1), 1.0 / 16, np.float32)
    G_pair = np.pi * (sAB @ sAB.T)
    G_cav = np.pi * (np.kron(PA, I) + np.kron(I, PA))
    G_bdry = np.zeros((256, 256), np.float32)
    for li in range(8):
        for ri in range(8):
            idx = [(16 * ((li << 1) | b3)) + ((c0 << 3) | ri) for b3 in range(2) for c0 in range(2)]
            s = np.zeros((256, 1), np.float32); s[idx, 0] = 0.5
            G_bdry += np.pi * (s @ s.T)
    return jnp.asarray(np.stack([G_pair, G_cav, G_bdry]))
PAIR_G = _grover_gen()
def pair_init(rng):
    return dict(pth=jnp.asarray(np.array([[0.00, 0.00, 0.00],
                                         [0.00, 0.00, 0.08],
                                         [0.00, 0.00, 0.10],
                                         [0.00, 0.00, 0.08]], np.float32)))
def pair_outputs(p):
    return dict(pHr=jnp.einsum("lk,kij->lij", p["pth"], PAIR_G)[:, None, None], pslots=PAIR_SLOTS)

def apply(p, xe, xo, inc, sgn, V, place=None):
    """3-round shared-weight MP. Sign flip of var i: e unchanged, o_i -> -o_i, pv_i -> 1-pv_i, gamma/beta/w fixed."""
    e = jnp.tanh(xe @ p["We"]); o = jnp.tanh(xo @ p["Wo"]); vd = inc.sum(1)[..., None] + 1.0
    for _ in range(4):
        me = jnp.einsum("fmn,fnh->fmh", inc, e); mo = jnp.einsum("fmn,fnh->fmh", sgn, o)  # |SG|·e even; SG·o even
        hc = jnp.tanh((me @ p["Wce"] + mo @ p["Wco"]) / 4.0)
        e = jnp.tanh(e + (jnp.einsum("fmn,fmh->fnh", inc, hc) / vd) @ p["Wve"])  # even via |SG|
        o = jnp.tanh(o + (jnp.einsum("fmn,fmh->fnh", sgn, hc) / vd) @ p["Wvo"])  # odd via SG
    hc = jnp.tanh(jnp.concatenate([jnp.einsum("fmn,fnh->fmh", inc, e), jnp.einsum("fmn,fnh->fmh", sgn, o)], -1) / 4.0 @ p["Wc"])
    if place is not None:
        gr, lost, cidx = geom(V, place, inc.shape[2]); geo = gr.mean(1)
        lv = jnp.einsum("frm,fmn->frn", lost, inc) / vd[:, None, :, 0]
        lvs = jnp.einsum("frm,fmn->frn", lost * gr[..., 3], sgn) / vd[:, None, :, 0]  # hop-2 only, signed
    else:
        geo = jnp.zeros(V.shape[:2] + (4,), jnp.float32); lost = cidx = lvs = None
    gb = jnp.tanh(e.mean(1) @ p["Wg"] + geo.mean(1) @ p["Wgg"])
    w = hc @ p["wc"] + jnp.tanh(jnp.concatenate([hc, geo], -1) @ p["Wq1"]) @ p["Wq2"] + geo[..., 3, None] * p["wsk"]
    bref = p["bref"] * jnp.asarray([0.0, 0.0, 0.0, 1.0], jnp.float32)  # L4 refocus only; 0 => parent β
    out = dict(gamma=p["g"][None] + gb[:, :4], beta=p["b"][None] + gb[:, 4:] + bref[None], w=w, pv=jax.nn.sigmoid(o @ p["wp"]))
    if place is not None:
        out["dS"], out["dB"] = corr(p, e, o, place, lv, lvs, lost, cidx)
        out["dS"], out["dB"] = native(p, hc, V, sgn, place, inc.shape[2], out["dS"], out["dB"], geo.mean(1)[:, :3])
    out.update(pair_outputs(p))
    return out

def train(api, budget_s):
    t0 = time.time(); rng = np.random.default_rng(0); p = init(rng); p.update(native_init(rng)); p.update(pair_init(rng))

    from jax.flatten_util import ravel_pytree
    th, unr = ravel_pytree(p); mo = jnp.zeros_like(th); ve = jnp.zeros_like(th)
    data = {n: api.data(n) for n in api.n_list}; feats = {n: features(n, d["V"], d["SG"], d["m"]) for n, d in data.items()}
    bs = {8: 12, 10: 8, 12: 12, 15: 4}
    def loss(th, n, fx, pl, b):
        pp = unr(th); out = apply(pp, *fx, place=pl)
        P = api.p_from(n, out, b); nl = -jnp.log(P.mean(1) + 1e-12)
        xe, xo, inc, sgn, V = fx; of = apply(pp, xe, -xo, inc, -sgn, V, place=pl)  # all-sign flip: pv->1-pv, even heads fixed
        pen = ((out["gamma"] - of["gamma"]) ** 2).mean() + ((out["beta"] - of["beta"]) ** 2).mean()
        pen = pen + ((out["w"] - of["w"]) ** 2).mean() + ((out["pv"] + of["pv"] - 1.0) ** 2).mean()
        _, Lst, _ = geom(V, pl, inc.shape[2])
        lw = 1.0 + 1.5 * jax.lax.stop_gradient(Lst.mean((1, 2)))  # upweight skip-clause formulas (screen tail)
        kc = max(1, int(np.ceil(0.1 * nl.shape[0])))   # CVaR-10% of log(1/p) (tail) + smooth median/mean terms
        tot = 0.5 * (nl * lw).mean() + 0.5 * jnp.median(nl) + TAIL_W * jax.lax.top_k(nl, kc)[0].mean() + 0.05 * pen
        if LAM_GLOB > 0:  # Bernoulli KL(p_glob || p_chain) per (formula, placement); global p is a stop-gradient target
            Pg = jax.lax.stop_gradient(jnp.clip(api.p_from(n, out, b, glob=True), 1e-6, 1 - 1e-6)); Pc = jnp.clip(P, 1e-6, 1 - 1e-6)
            tot = tot + LAM_GLOB * (Pg * jnp.log(Pg / Pc) + (1 - Pg) * jnp.log((1 - Pg) / (1 - Pc))).mean()
        return tot
    vg = {n: jax.jit(jax.value_and_grad(lambda th, fx, pl, b, n=n: loss(th, n, fx, {"cav": pl, "ks": api.place(n, [0])["ks"]}, b))) for n in api.n_list}
    t = 0; T = budget_s * 0.9; ema = None
    while time.time() - t0 < T:
        g = 0
        for n in api.n_list:
            idx = rng.choice(data[n]["F"], bs.get(n, 8), replace=False); fx = tuple(x[idx] for x in feats[n])
            _, gn = vg[n](th, fx, jnp.asarray(api.place(n, idx)["cav"]), api.batch(n, idx)); g = g + gn
        gnorm = float(jnp.sqrt((g * g).sum()))
        if not np.isfinite(gnorm): continue
        t += 1; g = g * min(1.0, 10.0 / (gnorm + 1e-9))
        fr = (time.time() - t0) / T; lr = 3e-3 * (1.0 if fr < 0.7 else max(0.2, (1.0 - fr) / 0.3))
        mo = 0.9 * mo + 0.1 * g; ve = 0.999 * ve + 0.001 * g * g
        th = th - lr * (mo / (1 - 0.9 ** t)) / (jnp.sqrt(ve / (1 - 0.999 ** t)) + 1e-8)
        if fr > 0.4: ema = th if ema is None else 0.98 * ema + 0.02 * th
    pf = unr(th if ema is None or not bool(jnp.all(jnp.isfinite(ema))) else ema)
    def predict(n, V, SG, m, place=None):
        V, SG = np.asarray(V), np.asarray(SG); res = []
        for s in range(0, V.shape[0], 32):
            pl = None if place is None else dict(cav=np.asarray(place["cav"])[s:s + 32], ks=place["ks"])
            out = apply(pf, *features(n, V[s:s + 32], SG[s:s + 32], m), place=pl)
            res.append({k: np.nan_to_num(np.asarray(v)) for k, v in out.items() if k not in ('pHr', 'pHi', 'pslots')})
        sh = pair_outputs(pf)
        return dict({k: np.concatenate([r[k] for r in res]) for k in res[0] if k not in sh}, **{k: (v if k == 'pslots' else np.asarray(v)) for k, v in sh.items()})
    return predict
# EVOLVE-BLOCK-END
